#!/usr/bin/env python3
"""
JSONL 数据集分割脚本
将单个 JSONL 文件按比例随机分割为训练集和验证集

功能：
- 流式读取，支持大文件（避免 OOM）
- 70% 训练集，30% 验证集
- 随机划分
- 输出 train.jsonl 和 valid.jsonl

使用方法：
    python split_dataset.py input.jsonl [--train-ratio 0.7] [--output-dir ./]
"""

import json
import random
import sys
import argparse
from pathlib import Path
from typing import Iterator, Tuple


def read_jsonl_stream(file_path: Path) -> Iterator[dict]:
    """
    流式读取 JSONL 文件，逐行返回 JSON 对象
    
    Args:
        file_path: JSONL 文件路径
        
    Yields:
        dict: 每行的 JSON 对象
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {line_num} 行 JSON 解析失败，跳过: {e}", file=sys.stderr)
                continue


def count_lines(file_path: Path) -> int:
    """
    快速统计文件行数（用于显示进度）
    
    Args:
        file_path: 文件路径
        
    Returns:
        int: 文件行数
    """
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 只统计非空行
                count += 1
    return count


def split_jsonl(
    input_file: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    seed: int = 42
) -> Tuple[Path, Path]:
    """
    将 JSONL 文件分割为训练集和验证集
    
    Args:
        input_file: 输入的 JSONL 文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例（默认 0.7，即 70%）
        seed: 随机种子（默认 42，确保可复现）
        
    Returns:
        Tuple[Path, Path]: (train_file, valid_file) 输出文件路径
    """
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件路径
    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"
    
    # 统计总行数（用于显示进度）
    print(f"正在统计文件行数...")
    total_lines = count_lines(input_file)
    print(f"总行数: {total_lines}")
    
    # 打开输出文件
    train_f = open(train_file, 'w', encoding='utf-8')
    valid_f = open(valid_file, 'w', encoding='utf-8')
    
    try:
        train_count = 0
        valid_count = 0
        
        print(f"\n开始分割数据集...")
        print(f"训练集比例: {train_ratio * 100:.1f}%")
        print(f"验证集比例: {(1 - train_ratio) * 100:.1f}%")
        print(f"随机种子: {seed}")
        print("-" * 60)
        
        # 流式读取并分割
        for line_num, data in enumerate(read_jsonl_stream(input_file), 1):
            # 生成随机数决定归属
            if random.random() < train_ratio:
                # 写入训练集
                train_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                train_count += 1
            else:
                # 写入验证集
                valid_f.write(json.dumps(data, ensure_ascii=False) + '\n')
                valid_count += 1
            
            # 每处理 1000 行显示一次进度
            if line_num % 1000 == 0:
                progress = (line_num / total_lines) * 100
                print(f"进度: {line_num}/{total_lines} ({progress:.1f}%) | "
                      f"训练集: {train_count} | 验证集: {valid_count}")
        
        # 最终统计
        print("-" * 60)
        print(f"分割完成！")
        print(f"训练集: {train_count} 条 ({train_count/total_lines*100:.1f}%)")
        print(f"验证集: {valid_count} 条 ({valid_count/total_lines*100:.1f}%)")
        print(f"\n输出文件:")
        print(f"  训练集: {train_file}")
        print(f"  验证集: {valid_file}")
        
    finally:
        train_f.close()
        valid_f.close()
    
    return train_file, valid_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将 JSONL 文件分割为训练集和验证集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用（70% 训练，30% 验证）
  python split_dataset.py data.jsonl
  
  # 指定输出目录
  python split_dataset.py data.jsonl --output-dir ./dataset
  
  # 自定义训练集比例（80% 训练，20% 验证）
  python split_dataset.py data.jsonl --train-ratio 0.8
  
  # 指定随机种子（确保可复现）
  python split_dataset.py data.jsonl --seed 123
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入的 JSONL 文件路径'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='训练集比例（默认: 0.7，即 70%%）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='输出目录（默认: 当前目录）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42，确保可复现）'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not (0 < args.train_ratio < 1):
        print("错误: train-ratio 必须在 0 和 1 之间", file=sys.stderr)
        sys.exit(1)
    
    # 检查输入文件
    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    if not input_file.is_file():
        print(f"错误: 输入路径不是文件: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    
    try:
        # 执行分割
        train_file, valid_file = split_jsonl(
            input_file=input_file,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            seed=args.seed
        )
        
        print(f"\n✅ 成功！")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  操作被用户中断", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\n❌ 发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

