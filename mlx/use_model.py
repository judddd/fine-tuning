#!/usr/bin/env python3
"""
MLX-LM 模型使用脚本
支持：命令行推理、批量推理、API 服务

使用方法：
1. 单次推理：python use_model.py --prompt "你的问题"
2. 交互模式：python use_model.py --interactive
3. 批量推理：python use_model.py --batch input.jsonl --output results.jsonl
4. API 服务：python use_model.py --serve --port 8080
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

def find_latest_adapter() -> Optional[Path]:
    """自动查找最新的适配器"""
    saves_dir = Path("saves/qwen-lora")
    if not saves_dir.exists():
        return None
    
    train_dirs = sorted(saves_dir.glob("train_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for train_dir in train_dirs:
        adapter_file = train_dir / "adapters.npz"
        if adapter_file.exists():
            return adapter_file
    
    return None

def load_model(model_name: str, adapter_path: Optional[str] = None):
    """加载模型和适配器"""
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("❌ 错误: 请先安装 mlx-lm")
        print("   pip install mlx-lm")
        sys.exit(1)
    
    print("🔄 加载模型...")
    print(f"   模型: {model_name}")
    
    if adapter_path:
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            print(f"❌ 适配器文件不存在: {adapter_path}")
            sys.exit(1)
        print(f"   适配器: {adapter_path}")
        model, tokenizer = load(model_name, adapter_path=str(adapter_path))
    else:
        print("   ⚠️  未使用适配器（基础模型）")
        model, tokenizer = load(model_name)
    
    print("✅ 模型加载完成\n")
    return model, tokenizer, generate

def single_inference(model, tokenizer, generate_fn, prompt: str, 
                     max_tokens: int = 500, temperature: float = 0.7) -> str:
    """单次推理"""
    response = generate_fn(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False
    )
    return response

def interactive_mode(model, tokenizer, generate_fn, 
                     max_tokens: int = 500, temperature: float = 0.7):
    """交互式对话模式"""
    print("=" * 60)
    print("🤖 MLX-LM 交互式问答")
    print("=" * 60)
    print("提示:")
    print("  - 输入你的问题，模型会给出回答")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 输入 'clear' 清空屏幕")
    print("=" * 60)
    print()
    
    while True:
        try:
            prompt = input("🧑 你> ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("\n👋 再见！")
                break
            
            if prompt.lower() == 'clear':
                import os
                os.system('clear' if sys.platform != 'win32' else 'cls')
                continue
            
            print("\n🤖 AI> ", end="", flush=True)
            response = single_inference(model, tokenizer, generate_fn, prompt, max_tokens, temperature)
            print(response)
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")

def batch_inference(model, tokenizer, generate_fn, input_file: str, output_file: str,
                    max_tokens: int = 500, temperature: float = 0.7):
    """批量推理"""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return
    
    print(f"📂 批量推理")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    print()
    
    results = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                prompt = item.get("prompt", item.get("question", ""))
                
                if not prompt:
                    print(f"⚠️  第 {i} 行: 未找到 'prompt' 或 'question' 字段，跳过")
                    continue
                
                print(f"处理 {i}: {prompt[:50]}...")
                response = single_inference(model, tokenizer, generate_fn, prompt, max_tokens, temperature)
                
                result = {
                    "prompt": prompt,
                    "response": response,
                    **{k: v for k, v in item.items() if k not in ["prompt", "question"]}
                }
                results.append(result)
                
            except Exception as e:
                print(f"❌ 第 {i} 行处理失败: {e}")
                continue
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 完成! 共处理 {len(results)} 条")
    print(f"   结果保存到: {output_path}")

def serve_api(model, tokenizer, generate_fn, port: int = 8080, 
              max_tokens: int = 500, temperature: float = 0.7):
    """启动 API 服务"""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("❌ 错误: 需要安装 Flask")
        print("   pip install flask flask-cors")
        sys.exit(1)
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok"})
    
    @app.route('/generate', methods=['POST'])
    def generate_endpoint():
        try:
            data = request.json
            prompt = data.get('prompt')
            if not prompt:
                return jsonify({"error": "Missing 'prompt' field"}), 400
            
            response = single_inference(
                model, tokenizer, generate_fn,
                prompt=prompt,
                max_tokens=data.get('max_tokens', max_tokens),
                temperature=data.get('temperature', temperature)
            )
            
            return jsonify({
                "prompt": prompt,
                "response": response
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/chat', methods=['POST'])
    def chat_endpoint():
        try:
            data = request.json
            messages = data.get('messages', [])
            if not messages:
                return jsonify({"error": "Missing 'messages' field"}), 400
            
            # 构建提示
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'system':
                    prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
                elif role == 'user':
                    prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                elif role == 'assistant':
                    prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(prompt_parts)
            
            response = single_inference(
                model, tokenizer, generate_fn,
                prompt=prompt,
                max_tokens=data.get('max_tokens', max_tokens),
                temperature=data.get('temperature', temperature)
            )
            
            return jsonify({
                "messages": messages + [{"role": "assistant", "content": response}],
                "response": response
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    print("=" * 60)
    print("🚀 MLX-LM API 服务器")
    print("=" * 60)
    print(f"   地址: http://localhost:{port}")
    print(f"   健康检查: http://localhost:{port}/health")
    print(f"   生成接口: POST http://localhost:{port}/generate")
    print(f"   对话接口: POST http://localhost:{port}/chat")
    print("=" * 60)
    print("\n按 Ctrl+C 停止服务\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)

def main():
    parser = argparse.ArgumentParser(
        description="MLX-LM 模型使用脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单次推理
  python use_model.py --prompt "什么是 Elasticsearch"
  
  # 交互模式
  python use_model.py --interactive
  
  # 批量推理
  python use_model.py --batch questions.jsonl --output answers.jsonl
  
  # API 服务
  python use_model.py --serve --port 8080
  
  # 使用特定适配器
  python use_model.py --adapter saves/xxx/adapters.npz --interactive
  
  # 不使用适配器（基础模型）
  python use_model.py --no-adapter --prompt "测试"
        """
    )
    
    parser.add_argument('--model', type=str, 
                       default='mlx-community/Qwen2.5-3B-Instruct-4bit',
                       help='模型名称或路径')
    parser.add_argument('--adapter', type=str, default=None,
                       help='适配器路径 (默认: 自动查找最新)')
    parser.add_argument('--no-adapter', action='store_true',
                       help='不使用适配器（使用基础模型）')
    
    # 模式选择
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--prompt', type=str, help='单次推理提示')
    mode_group.add_argument('--interactive', '-i', action='store_true',
                           help='交互式模式')
    mode_group.add_argument('--batch', type=str, help='批量推理输入文件')
    mode_group.add_argument('--serve', action='store_true', help='启动 API 服务')
    
    # 推理参数
    parser.add_argument('--max-tokens', type=int, default=500,
                       help='最大生成 token 数 (默认: 500)')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='温度参数 (默认: 0.7)')
    
    # 批量推理参数
    parser.add_argument('--output', type=str, help='批量推理输出文件')
    
    # API 服务参数
    parser.add_argument('--port', type=int, default=8080,
                       help='API 服务端口 (默认: 8080)')
    
    args = parser.parse_args()
    
    # 确定适配器路径
    adapter_path = None
    if not args.no_adapter:
        if args.adapter:
            adapter_path = args.adapter
        else:
            adapter_path = find_latest_adapter()
            if adapter_path:
                print(f"🔍 自动找到适配器: {adapter_path}\n")
            else:
                print("⚠️  未找到适配器，将使用基础模型")
                print("   提示: 使用 --adapter 指定适配器路径\n")
    
    # 加载模型
    model, tokenizer, generate_fn = load_model(args.model, adapter_path)
    
    # 执行对应模式
    if args.prompt:
        response = single_inference(model, tokenizer, generate_fn, 
                                   args.prompt, args.max_tokens, args.temperature)
        print("=" * 60)
        print("问题:")
        print(args.prompt)
        print("\n回答:")
        print(response)
        print("=" * 60)
    
    elif args.interactive:
        interactive_mode(model, tokenizer, generate_fn, 
                        args.max_tokens, args.temperature)
    
    elif args.batch:
        if not args.output:
            print("❌ 批量推理需要指定 --output 参数")
            sys.exit(1)
        batch_inference(model, tokenizer, generate_fn, 
                       args.batch, args.output, args.max_tokens, args.temperature)
    
    elif args.serve:
        serve_api(model, tokenizer, generate_fn, args.port, 
                 args.max_tokens, args.temperature)

if __name__ == "__main__":
    main()

