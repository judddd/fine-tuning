"""
模型管理器 - 加载和管理本地LLM模型
支持 Hugging Face 格式的本地模型
"""

import os
import json
import logging
from typing import Generator, Optional, Dict, Any
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    GenerationConfig
)
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_model_name_from_path(model_path: str) -> str:
    """
    从模型路径自动检测模型名称
    
    优先级:
    1. config.json 中的 model_name 或 _name_or_path
    2. 路径中的最后两级目录 (如 Qwen/Qwen3-Next-80B-A3B-Instruct)
    3. 路径中的最后一级目录
    
    Args:
        model_path: 模型本地路径
        
    Returns:
        检测到的模型名称
    """
    try:
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
        # 回退到路径最后一级
        return Path(model_path).name


class ModelManager:
    """本地模型管理器"""
    
    def __init__(self, model_path: str, model_name: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            model_path: 模型本地路径
            model_name: 模型显示名称（可选，不提供则自动检测）
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> bool:
        """
        加载模型到内存
        
        Returns:
            是否加载成功
        """
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            # 检查路径是否存在
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                return False
            
            # 加载 tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 设置 pad_token（如果没有）
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"✅ Model '{self.model_name}' loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def unload_model(self):
        """卸载模型释放内存"""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Model '{self.model_name}' unloaded")
    
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
        流式生成文本
        
        Args:
            prompt: 输入提示词
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: nucleus sampling参数
            
        Yields:
            生成的文本片段
        """
        if not self.is_loaded():
            yield "[ERROR] Model not loaded. Please load the model first."
            return
        
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # 创建流式输出器
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # 生成配置
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # 在后台线程中生成
            generation_kwargs = dict(
                **inputs,
                generation_config=generation_config,
                streamer=streamer,
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # 流式输出
            for text in streamer:
                yield text
            
            thread.join()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield f"\n\n[ERROR] {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_loaded():
            return {
                "name": self.model_name,
                "path": self.model_path,
                "loaded": False,
                "device": self.device
            }
        
        # 计算参数量
        num_parameters = sum(p.numel() for p in self.model.parameters())
        
        return {
            "name": self.model_name,
            "path": self.model_path,
            "loaded": True,
            "device": str(self.model.device) if hasattr(self.model, 'device') else self.device,
            "num_parameters": f"{num_parameters:,}",
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown"
        }

