"""
模型管理器 - 加载和管理本地LLM模型（MLX 版本）
支持 MLX-LM 格式的本地模型和 LoRA 适配器
"""

import os
import json
import logging
from typing import Generator, Optional, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_saves_path(saves_dir: str = "mlx/saves/qwen-lora") -> Optional[Path]:
    """
    解析适配器保存目录路径
    
    Args:
        saves_dir: 保存目录路径（相对于项目根目录或绝对路径）
        
    Returns:
        解析后的路径，如果不存在则返回 None
    """
    saves_path = Path(saves_dir)
    # 如果是相对路径，尝试从当前工作目录和脚本目录查找
    if not saves_path.is_absolute():
        # 先尝试当前工作目录
        if not saves_path.exists():
            # 尝试从脚本所在目录（fine-tune 目录）
            script_dir = Path(__file__).parent
            saves_path = script_dir / saves_dir
            if not saves_path.exists():
                # 尝试从项目根目录
                project_root = script_dir.parent if script_dir.name == "fine-tune" else script_dir
                saves_path = project_root / saves_dir
    
    if not saves_path.exists():
        logger.warning(f"适配器保存目录不存在: {saves_path}")
        return None
    
    return saves_path


def find_latest_adapter(saves_dir: str = "mlx/saves/qwen-lora") -> Optional[Path]:
    """
    自动查找最新的适配器
    
    Args:
        saves_dir: 保存目录路径（相对于项目根目录或绝对路径）
        
    Returns:
        最新的适配器文件路径，如果不存在则返回 None
    """
    saves_path = resolve_saves_path(saves_dir)
    if saves_path is None:
        return None
    
    # 查找所有 train_* 目录
    train_dirs = sorted(
        saves_path.glob("train_*"), 
        key=lambda p: p.stat().st_mtime, 
        reverse=True
    )
    
    for train_dir in train_dirs:
        adapter_file = train_dir / "adapters.npz"
        if adapter_file.exists():
            logger.info(f"找到适配器: {adapter_file}")
            return adapter_file
    
    logger.warning("未找到任何适配器文件")
    return None


def load_adapter_config(adapter_path: Path) -> Optional[Dict[str, Any]]:
    """
    加载适配器配置文件
    
    Args:
        adapter_path: 适配器文件路径（adapters.npz）
        
    Returns:
        配置字典，如果文件不存在则返回 None
    """
    # adapter_config.json 在 adapters.npz 的同一目录下
    config_file = adapter_path.parent / "adapter_config.json"
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"无法读取适配器配置 {config_file}: {e}")
        return None


def extract_adapter_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从适配器配置中提取重要信息
    
    Args:
        config: 适配器配置字典
        
    Returns:
        提取的重要信息字典
    """
    info = {}
    
    # LoRA 参数
    if "lora_parameters" in config:
        lora = config["lora_parameters"]
        info["lora_rank"] = lora.get("rank", "N/A")
        info["lora_scale"] = lora.get("scale", "N/A")
        info["lora_dropout"] = lora.get("dropout", "N/A")
    
    # 训练参数
    info["learning_rate"] = config.get("learning_rate", "N/A")
    info["batch_size"] = config.get("batch_size", "N/A")
    info["iters"] = config.get("iters", "N/A")
    info["num_layers"] = config.get("num_layers", "N/A")
    info["max_seq_length"] = config.get("max_seq_length", "N/A")
    
    # 其他重要参数
    info["grad_checkpoint"] = config.get("grad_checkpoint", False)
    info["optimizer"] = config.get("optimizer", "N/A")
    
    return info


def list_all_adapters(saves_dir: str = "mlx/saves/qwen-lora") -> list:
    """
    列出所有可用的适配器（按时间排序，最新的在前）
    
    Args:
        saves_dir: 保存目录路径（相对于项目根目录或绝对路径）
        
    Returns:
        适配器列表，每个元素包含：
        - path: 适配器文件路径（字符串）
        - name: 适配器名称（train_YYYY-MM-DD-HH-MM-SS）
        - mtime: 修改时间戳
        - config: 适配器配置信息（如果存在）
    """
    saves_path = resolve_saves_path(saves_dir)
    if saves_path is None:
        return []
    
    adapters = []
    # 查找所有 train_* 目录
    train_dirs = sorted(
        saves_path.glob("train_*"), 
        key=lambda p: p.stat().st_mtime, 
        reverse=True  # 最新的在前
    )
    
    for train_dir in train_dirs:
        adapter_file = train_dir / "adapters.npz"
        if adapter_file.exists():
            adapter_info = {
                "path": str(adapter_file),
                "name": train_dir.name,
                "mtime": adapter_file.stat().st_mtime
            }
            
            # 尝试加载配置信息
            config = load_adapter_config(adapter_file)
            if config:
                adapter_info["config"] = extract_adapter_info(config)
            
            adapters.append(adapter_info)
    
    return adapters


def detect_model_name_from_path(model_path: str) -> str:
    """
    从模型路径自动检测模型名称
    
    优先级:
    1. config.json 中的 model_name 或 _name_or_path
    2. 路径中的最后两级目录 (如 Qwen/Qwen3-Next-80B-A3B-Instruct)
    3. 路径中的最后一级目录
    
    Args:
        model_path: 模型本地路径或 HuggingFace 模型 ID
        
    Returns:
        检测到的模型名称
    """
    try:
        # 如果是 HuggingFace 模型 ID（包含 /），直接返回
        if "/" in model_path and not Path(model_path).exists():
            return model_path
        
        # 尝试从 config.json 读取
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 优先使用 _name_or_path
                if "_name_or_path" in config and config["_name_or_path"]:
                    name = config["_name_or_path"]
                    # 如果是本地路径，提取最后两级
                    if "/" in name or "\\" in name:
                        parts = Path(name).parts
                        if len(parts) >= 2:
                            return f"{parts[-2]}/{parts[-1]}"
                        return parts[-1]
                    return name
                
                # 尝试 model_name
                if "model_name" in config and config["model_name"]:
                    return config["model_name"]
        
        # 从路径提取最后两级 (如 Qwen/Qwen3-Next-80B-A3B-Instruct)
        path_obj = Path(model_path)
        parts = path_obj.parts
        
        if len(parts) >= 2:
            # 检查是否是常见的模型仓库结构
            if parts[-2] in ["models", "hub", "model"]:
                # 如果倒数第二级是 models/hub，再往前取
                if len(parts) >= 3:
                    return f"{parts[-3]}/{parts[-1]}"
            else:
                return f"{parts[-2]}/{parts[-1]}"
        
        # 只返回最后一级
        return parts[-1]
        
    except Exception as e:
        logger.warning(f"Failed to detect model name from config: {e}")
        # 回退到路径最后一级或原路径
        return Path(model_path).name if Path(model_path).exists() else model_path


class ModelManager:
    """本地模型管理器（MLX 版本）"""
    
    def __init__(
        self, 
        model_path: str, 
        model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        no_adapter: bool = False,
        saves_dir: str = "mlx/saves/qwen-lora"
    ):
        """
        初始化模型管理器
        
        Args:
            model_path: 模型路径或 HuggingFace 模型 ID
            model_name: 模型显示名称（可选，不提供则自动检测）
            adapter_path: 适配器文件路径（可选，不提供则自动查找最新的）
            no_adapter: 是否不使用适配器（使用基础模型）
            saves_dir: 适配器保存目录（用于自动查找）
        """
        self.model_path = model_path
        # 如果未提供名称，自动检测
        if model_name is None or model_name.strip() == "":
            self.model_name = detect_model_name_from_path(model_path)
            logger.info(f"Auto-detected model name: {self.model_name}")
        else:
            self.model_name = model_name
        
        self.model = None
        self.tokenizer = None
        self.generate_fn = None
        
        # 适配器配置
        self.no_adapter = no_adapter
        self.saves_dir = saves_dir
        
        # 确定适配器路径
        if no_adapter:
            self.adapter_path = None
            logger.info("配置为不使用适配器（基础模型）")
        elif adapter_path:
            self.adapter_path = Path(adapter_path)
            logger.info(f"使用指定的适配器: {self.adapter_path}")
        else:
            # 自动查找最新的适配器
            self.adapter_path = find_latest_adapter(saves_dir)
            if self.adapter_path:
                logger.info(f"自动找到适配器: {self.adapter_path}")
            else:
                logger.warning("未找到适配器，将使用基础模型")
        
        # 检查 MLX-LM 是否可用
        try:
            from mlx_lm import load, generate
            self._mlx_available = True
            logger.debug("MLX-LM 导入成功")
        except ImportError as e:
            self._mlx_available = False
            logger.error(f"MLX-LM 导入失败 (ImportError): {e}")
            logger.error("请运行: pip install mlx-lm")
        except Exception as e:
            self._mlx_available = False
            logger.error(f"MLX-LM 导入失败 (其他错误): {type(e).__name__}: {e}")
            logger.error("请检查 MLX-LM 是否正确安装: pip install mlx-lm")
    
    def load_model(self) -> bool:
        """
        加载模型到内存（MLX 方式）
        
        Returns:
            是否加载成功
        """
        # 再次尝试导入（以防运行时环境变化）
        try:
            from mlx_lm import load, generate
        except ImportError as e:
            logger.error(f"MLX-LM 导入失败: {e}")
            logger.error("请确保已安装 MLX-LM: pip install mlx-lm")
            logger.error("如果已安装，请检查 Python 环境是否正确")
            return False
        except Exception as e:
            logger.error(f"MLX-LM 导入时发生错误: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
        if not self._mlx_available:
            logger.warning("初始化时 MLX-LM 不可用，但当前导入成功，继续加载...")
        
        try:
            logger.info(f"🔄 加载模型: {self.model_path}")
            
            # 检查适配器路径（如果指定了）
            if self.adapter_path and not self.adapter_path.exists():
                logger.error(f"适配器文件不存在: {self.adapter_path}")
                return False
            
            # 加载模型和适配器
            if self.adapter_path:
                logger.info(f"   使用适配器: {self.adapter_path}")
                self.model, self.tokenizer = load(
                    self.model_path, 
                    adapter_path=str(self.adapter_path)
                )
            else:
                logger.info("   ⚠️  未使用适配器（基础模型）")
                self.model, self.tokenizer = load(self.model_path)
            
            # 保存 generate 函数引用
            self.generate_fn = generate
            
            logger.info(f"✅ 模型 '{self.model_name}' 加载成功!")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def unload_model(self):
        """卸载模型释放内存"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if self.generate_fn:
            self.generate_fn = None
        
        logger.info(f"模型 '{self.model_name}' 已卸载")
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.tokenizer is not None
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成文本（MLX 方式）
        
        注意：MLX-LM 的 generate 函数不支持 max_tokens、temperature、top_p 等参数
        这些参数会被忽略，使用模型的默认设置
        
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数（MLX 不支持，参数保留以兼容 API）
            temperature: 温度参数（MLX 不支持，参数保留以兼容 API）
            top_p: nucleus sampling参数（MLX 不支持，参数保留以兼容 API）
            
        Yields:
            生成的文本片段
        """
        if not self.is_loaded():
            yield "[ERROR] 模型未加载，请先加载模型"
            return
        
        try:
            # MLX-LM 的 generate 函数只支持 prompt 和 verbose 参数
            # 参考 use_model.py 的实现，不传递 max_tokens、temperature 等参数
            response = self.generate_fn(
                self.model,
                self.tokenizer,
                prompt=prompt,
                verbose=False
            )
            
            # 将完整响应分块返回（模拟流式输出）
            # 可以按字符或按词分块
            chunk_size = 5  # 每次返回 5 个字符
            for i in range(0, len(response), chunk_size):
                chunk = response[i:i + chunk_size]
                yield chunk
            
        except Exception as e:
            logger.error(f"生成错误: {e}")
            import traceback
            traceback.print_exc()
            yield f"\n\n[ERROR] {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "name": self.model_name,
            "path": self.model_path,
            "loaded": self.is_loaded(),
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "using_adapter": self.adapter_path is not None and not self.no_adapter,
            "framework": "MLX"
        }
        
        # 如果使用了适配器，尝试加载适配器配置信息
        if self.adapter_path and self.adapter_path.exists():
            config = load_adapter_config(self.adapter_path)
            if config:
                adapter_info = extract_adapter_info(config)
                info["adapter_config"] = adapter_info
        
        if self.is_loaded():
            info["device"] = "Apple Silicon (MLX)"
        
        return info

