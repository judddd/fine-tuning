#!/usr/bin/env python3
"""
MLX 训练脚本 - 使用 YAML 配置文件
通过生成 config.yaml 并使用 --config 参数传递所有配置

快速开始：
1. 准备数据：将训练数据放在 ./data 目录下（train.jsonl）
2. 选择模型：在 config 中修改 "model" 参数
   - 使用 HuggingFace 模型 ID（推荐，自动下载）
   - 或使用本地模型路径
3. 运行：python train.py

支持的模型格式：
- HuggingFace 模型 ID: "mlx-community/Qwen2.5-3B-Instruct-4bit"
- 本地路径: "/path/to/model" (需包含 config.json 和权重文件)

配置方式：
- 脚本会自动生成 train_config.yaml 文件
- 所有 LoRA 参数（lora_rank, lora_alpha, lora_dropout）都会被正确使用
- 使用 --config 参数传递给 MLX-LM

依赖：
- PyYAML: pip install PyYAML
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# 尝试导入 yaml，如果没有安装则提示
try:
    import yaml
except ImportError:
    print("错误: 需要安装 PyYAML 库")
    print("请运行: pip install PyYAML")
    sys.exit(1)

# ================== 配置参数 ==================
# 选项 1: HuggingFace 大模型（需要大内存）
# "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",


# 选项 2: 本地模型路径（确保路径存在且包含 config.json 和权重文件）
# "model": "/path/to/your/local/model",

config = {
    # 模型路径（使用 mlx-community 的 4bit 量化模型）
    "model": "/Users/newmind/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",  # 3B 模型，适合大多数 Mac
    
    # 数据目录
    "data": "../dataset",
    
    # LoRA 参数
    "num_layers": 16,  # 改名：lora_layers -> num_layers
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    
    # 训练参数
    "batch_size": 16,
    "iters": 1000, # （iters*batch/训练数据量=想要训练轮数） 总训练样本数 = 18750 × 16 = 300,000 个样本，如果你的数据只有 1000 条，每条会被训练 300 次
    "learning_rate": 5e-5,
    "max_seq_length": 2048,
    
    # 日志和保存
    "steps_per_report": 5,
    "steps_per_eval": 100,
    "save_every": 100,
    "val_batches": 25,
    
    # 输出目录（自动根据模型名称和时间戳生成）
    "output_dir": f"saves/qwen-lora/train_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
    
    # ============ 精度和优化 ============
    # 混合精度训练说明：
    # - MLX 框架默认使用 bfloat16 混合精度（自动启用，无需配置）
    # - 精度主要通过模型量化级别控制（4bit/8bit）
    # - 4bit 模型：更省内存，速度更快，精度略低
    # - 8bit 模型：平衡内存和精度
    # - 16bit (fp16/bfloat16)：最高精度，内存占用最大
    # 
    # 当前模型已使用量化（4bit/8bit），混合精度自动生效
    # 如需更高精度，使用未量化的模型或降低量化级别
    
    # 其他
    "grad_checkpoint": True,  # 梯度检查点（节省内存，稍慢）
    "seed": 42,
    "test": False,
}

# ================== 辅助函数 ==================

def get_script_dir():
    """获取脚本所在目录的绝对路径"""
    return Path(__file__).parent.absolute()

def resolve_path(path_str: str) -> Path:
    """将相对路径解析为基于脚本目录的绝对路径"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    # 相对路径基于脚本所在目录
    return get_script_dir() / path

# ================== 构建命令 ==================

def build_command():
    """构建 MLX 训练命令（使用 YAML 配置文件）"""
    
    # 解析数据目录为绝对路径
    data_path = resolve_path(config["data"])
    
    # 创建输出目录
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adapter_file = output_dir / "adapters.npz"
    
    # 构建 MLX-LM 训练配置（YAML 格式）
    # 注意：MLX-LM 使用 lora_parameters 结构，而不是直接的 lora_rank/lora_alpha
    mlx_config = {
        "model": config["model"],
        "data": str(data_path),
        "train": True,
        "adapter_path": str(adapter_file),
        "batch_size": config["batch_size"],
        "iters": config["iters"],
        "learning_rate": config["learning_rate"],
        "num_layers": config["num_layers"],
        "max_seq_length": config["max_seq_length"],
        "steps_per_report": config["steps_per_report"],
        "steps_per_eval": config["steps_per_eval"],
        "save_every": config["save_every"],
        "val_batches": config["val_batches"],
        "seed": config["seed"],
    }
    
    # LoRA 参数使用 lora_parameters 结构
    # scale = alpha / rank (MLX-LM 使用 scale 而不是 alpha)
    lora_scale = config["lora_alpha"] / config["lora_rank"] if config["lora_rank"] > 0 else 1.0
    mlx_config["lora_parameters"] = {
        "rank": config["lora_rank"],
        "scale": lora_scale,  # scale = alpha / rank
        "dropout": config["lora_dropout"],
    }
    
    # 添加可选参数
    if config["grad_checkpoint"]:
        mlx_config["grad_checkpoint"] = True
    
    if config["test"]:
        mlx_config["test"] = True
    
    # 保存 YAML 配置文件
    yaml_config_file = output_dir / "train_config.yaml"
    with open(yaml_config_file, "w", encoding="utf-8") as f:
        yaml.dump(mlx_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # 同时保存 JSON 配置（用于记录）
    json_config_file = output_dir / "config.json"
    with open(json_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("=" * 60)
    print("MLX 训练配置")
    print("=" * 60)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("=" * 60)
    print(f"数据目录: {data_path}")
    print(f"YAML 配置文件: {yaml_config_file}")
    print(f"JSON 配置（记录）: {json_config_file}")
    print(f"适配器将保存到: {adapter_file}")
    print("=" * 60)
    print("\nYAML 配置内容:")
    print("-" * 60)
    with open(yaml_config_file, "r", encoding="utf-8") as f:
        print(f.read())
    print("-" * 60)
    
    # 构建命令行参数（使用 --config 参数）
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", str(yaml_config_file),
    ]
    
    return cmd, adapter_file

# ================== 主函数 ==================

def main():
    """执行训练"""
    
    # 解析数据目录为绝对路径（基于脚本所在目录）
    data_path = resolve_path(config["data"])
    
    print(f"脚本目录: {get_script_dir()}")
    print(f"数据目录配置: {config['data']}")
    print(f"解析后的数据目录: {data_path}")
    print(f"数据目录是否存在: {data_path.exists()}")
    
    # 检查数据目录
    if not data_path.exists():
        print(f"错误: 数据目录不存在: {data_path}")
        print("\n请确保数据目录存在并包含以下文件之一:")
        print("  - train.jsonl")
        print("  - train.json")
        print("  - data.jsonl")
        return 1
    
    # 检查训练文件
    train_files = list(data_path.glob("*.jsonl")) + list(data_path.glob("train.json"))
    if not train_files:
        print(f"错误: 数据目录中没有找到训练文件")
        print(f"目录内容: {list(data_path.iterdir())}")
        print("\n支持的文件格式:")
        print("  - train.jsonl")
        print("  - *.jsonl")
        print("  - train.json")
        return 1
    
    print(f"找到训练文件: {[f.name for f in train_files]}")
    
    # 构建命令
    cmd, adapter_file = build_command()
    
    print("\n执行命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    # 执行训练
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"适配器已保存到: {adapter_file}")
        print(f"输出目录: {config['output_dir']}")
        
        # 如果需要测试
        if config["test"]:
            print("\n测试模型...")
            test_model(config["model"], str(adapter_file))
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        return 130
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

# ================== 测试函数 ==================

def test_model(model_path, adapter_path):
    """训练后测试模型"""
    try:
        from mlx_lm import load, generate
        
        print("\n加载模型...")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        
        # 测试提示
        test_prompts = [
            "User: 你好\nAssistant:",
            "User: 什么是机器学习？\nAssistant:",
            "User: 用Python写一个Hello World\nAssistant:",
        ]
        
        print("\n" + "=" * 60)
        print("模型测试")
        print("=" * 60)
        
        for prompt in test_prompts:
            print(f"\n提示: {prompt}")
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=100,
                temp=0.7
            )
            print(f"回答: {response}")
            print("-" * 60)
            
    except Exception as e:
        print(f"测试失败: {e}")

# ================== 使用说明 ==================

def print_usage():
    """打印使用说明"""
    print("""
使用方法:

1. 基本使用:
   python train.py
   
2. 选择模型 (修改脚本中的 config["model"]):
   
   # 推荐：使用 HuggingFace 模型（自动下载）
   "model": "mlx-community/Qwen2.5-3B-Instruct-4bit"
   
   # 其他可用模型：
   - mlx-community/Qwen2.5-0.5B-Instruct-4bit  (最小，~500MB)
   - mlx-community/Qwen2.5-1.5B-Instruct-4bit  (小，~1GB)
   - mlx-community/Qwen2.5-3B-Instruct-4bit    (推荐，~2GB)
   - mlx-community/Qwen2.5-7B-Instruct-4bit    (大，~4GB)
   - mlx-community/Qwen2.5-14B-Instruct-4bit   (很大，~8GB)
   
   # 使用本地模型：
   "model": "/path/to/your/local/model"  # 需要包含 config.json 和权重
   
3. 自定义训练参数:
   config = {
       "model": "模型路径或 HF ID",
       "data": "数据目录",
       "batch_size": 16,
       "iters": 18750,
       "num_layers": 16,  # LoRA 层数
       ...
   }

4. 数据准备:
   确保数据目录包含以下格式之一的文件:
   
   - train.jsonl (推荐):
     {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}
     
   - train.json:
     [{"messages": [...]}, ...]
     
   格式示例:
   {"text": "User: 你好\\nAssistant: 你好！"}
   或
   {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}

5. 查看训练进度:
   训练过程中会显示损失值和其他指标
   
6. 训练完成后:
   适配器文件保存在: saves/.../adapters.npz
   可以使用以下代码加载:
   
   from mlx_lm import load, generate
   model, tokenizer = load("模型路径", adapter_path="adapters.npz")
   response = generate(model, tokenizer, prompt="...")

常见问题:

Q: 内存不足怎么办？
A: 减小 batch_size，或添加 --quantization-bits 4

Q: 训练太慢？
A: 检查是否使用了 Apple Silicon GPU (M1/M2/M3)

Q: 数据格式错误？
A: 使用 convert_data.py 转换数据格式
""")

# ================== 入口点 ==================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print_usage()
        sys.exit(0)
    
    exit_code = main()
    sys.exit(exit_code)