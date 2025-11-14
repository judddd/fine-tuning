#!/bin/bash

# Fine-tune Model Comparison Tool - 环境测试脚本

echo "🔍 Testing Fine-tune Model Comparison Tool Setup..."
echo "=" * 60

# 检查项目目录
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Not in fine-tune directory"
    exit 1
fi
echo "✅ Project directory: OK"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "✅ Python: $PYTHON_VERSION"

# 检查 uv
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found"
    echo "   Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
UV_VERSION=$(uv --version)
echo "✅ uv: $UV_VERSION"

# 检查虚拟环境
if [ -d ".venv" ]; then
    echo "✅ Virtual environment exists"
else
    echo "⚠️  Virtual environment not found (will be created on first start)"
fi

# 检查必要文件
FILES=(
    "main.py"
    "model_manager.py"
    "static/index.html"
    "README.md"
    "start.sh"
)

echo ""
echo "📁 Checking project files..."
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - MISSING"
        exit 1
    fi
done

# 检查文档
DOC_FILE="/Users/xuhao/work/es/newsoft/docmanage/20251114_fine_tune_comparison_tool.md"
if [ -f "$DOC_FILE" ]; then
    echo "  ✅ Design documentation"
else
    echo "  ⚠️  Design documentation not found at $DOC_FILE"
fi

echo ""
echo "=" * 60
echo "✅ All checks passed!"
echo ""
echo "Next steps:"
echo "  1. Run: ./start.sh"
echo "  2. Open: http://localhost:8100"
echo "  3. Load your models and start comparing!"
echo ""
echo "Need help? Check README.md or QUICKSTART.md"
echo "=" * 60

