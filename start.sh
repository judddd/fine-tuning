#!/bin/bash

# Model Comparison Tool - 启动脚本

set -e

echo "🚀 Starting Model Comparison Tool..."
echo "=" * 60

# 检查是否在项目目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from the fine-tune directory."
    exit 1
fi

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed."
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi

# 安装依赖
echo "📦 Installing dependencies..."
uv pip install -e .

# 激活虚拟环境并启动服务
echo ""
echo "=" * 60
echo "✅ Setup complete!"
echo "🌐 Starting server on http://localhost:8100"
echo "📚 API Docs: http://localhost:8100/docs"
echo "=" * 60
echo ""

# 启动 FastAPI 服务
source .venv/bin/activate
python main.py

