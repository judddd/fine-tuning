# Model Comparison Tool

Fine-tuned Model Side-by-Side Comparison Tool - 模型对比工具

## 功能特性

- ✅ 支持加载两个本地 Hugging Face 格式的模型
- ✅ 流式输出，实时对比两个模型的响应
- ✅ 双列并排显示，方便观察差异
- ✅ 支持自定义生成参数（Temperature, Top-P, Max Tokens）
- ✅ 响应时间统计
- ✅ 现代化的 Web 界面

## 系统要求

- Python 3.11+
- uv (Python 包管理器)
- CUDA (可选，用于 GPU 加速)

## 快速开始

### 1. 安装依赖

使用 uv 管理依赖（推荐）：

```bash
cd fine-tune

# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate  # Windows

# 安装依赖
uv pip install -e .
```

### 2. 启动服务

```bash
# 确保虚拟环境已激活
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8100 --reload
```

### 3. 访问界面

打开浏览器访问：http://localhost:8100

## 使用说明

### 步骤 1: 加载模型

1. 在 **Model A** 区域填写：
   - **Model Name**: 原始模型名称（可选）
   - **Model Path**: 原始模型的本地路径（绝对路径）
   - 点击 **Load Model** 加载模型

2. 在 **Model B** 区域填写：
   - **Model Name**: Fine-tuned 模型名称（可选）
   - **Model Path**: Fine-tuned 模型的本地路径（绝对路径）
   - 点击 **Load Model** 加载模型

> **提示**：模型路径示例
> - macOS/Linux: `/Users/username/models/llama-2-7b`
> - Windows: `C:\Users\username\models\llama-2-7b`

### 步骤 2: 输入问题

在 **Input Prompt** 区域输入你想问的问题，例如：

```
请解释一下机器学习中的过拟合问题。
```

### 步骤 3: 调整参数（可选）

- **Max Tokens**: 生成的最大 token 数量（默认 512）
- **Temperature**: 控制随机性，值越高越随机（默认 0.7）
- **Top P**: Nucleus sampling 参数（默认 0.9）

### 步骤 4: 生成对比

点击 **🚀 Generate Comparison** 按钮，两个模型将同时开始生成响应。

你可以实时看到：
- 左侧：Model A 的响应
- 右侧：Model B 的响应
- 每个模型的响应时间

## 模型路径格式

支持 Hugging Face Transformers 格式的模型，目录结构应包含：

```
your-model/
├── config.json
├── tokenizer_config.json
├── tokenizer.json
├── pytorch_model.bin (或 model.safetensors)
└── ...
```

### 示例：使用 Hugging Face 下载的模型

```bash
# 下载模型到本地
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
save_path = "/Users/username/models/llama-2-7b"

# 下载并保存
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

然后在界面中填写 `/Users/username/models/llama-2-7b` 作为模型路径。

## API 文档

启动服务后，访问 http://localhost:8100/docs 查看完整的 API 文档（Swagger UI）。

### 主要 API 端点

- `POST /api/models/{model_id}/load` - 加载模型
- `POST /api/models/{model_id}/unload` - 卸载模型
- `GET /api/models/{model_id}/status` - 获取模型状态
- `POST /api/generate/stream` - 流式生成文本
- `GET /api/status` - 获取所有模型状态

## 性能优化建议

### GPU 加速

如果你有 NVIDIA GPU 和 CUDA：

```bash
# 安装 CUDA 版本的 PyTorch
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 内存优化

对于大模型，可以使用量化：

```bash
# 安装 bitsandbytes
uv pip install bitsandbytes

# 修改 model_manager.py，添加 8-bit 量化
load_in_8bit=True
```

### CPU 优化

如果只使用 CPU，建议使用较小的模型（7B 以下）或量化版本。

## 故障排查

### 问题 1: 模型加载失败

**错误**: `Model path does not exist`

**解决**:
- 确认模型路径正确
- 使用绝对路径
- 检查目录是否包含必要的模型文件

### 问题 2: 内存不足

**错误**: `CUDA out of memory` 或 Python 内存错误

**解决**:
- 使用较小的模型
- 启用量化 (8-bit 或 4-bit)
- 一次只加载一个模型
- 减少 `max_new_tokens` 参数

### 问题 3: 生成速度慢

**解决**:
- 使用 GPU 而非 CPU
- 使用较小的模型
- 减少 `max_new_tokens`
- 检查其他程序是否占用资源

## 项目结构

```
fine-tune/
├── pyproject.toml           # uv 依赖配置
├── main.py                  # FastAPI 后端主应用
├── model_manager.py         # 模型加载和推理管理
├── static/
│   └── index.html          # Web 前端界面
├── README.md               # 本文档
└── .venv/                  # 虚拟环境（自动创建）
```

## 技术栈

- **Backend**: FastAPI + Uvicorn
- **Model Loading**: Hugging Face Transformers
- **Streaming**: Server-Sent Events (SSE)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Package Manager**: uv

## 开发说明

### 修改代码后重启服务

如果使用 `--reload` 模式，修改 Python 文件后会自动重启：

```bash
uvicorn main:app --host 0.0.0.0 --port 8100 --reload
```

### 查看日志

服务日志会输出到控制台，包括：
- 模型加载状态
- 请求处理信息
- 错误信息

### 自定义端口

修改 `main.py` 最后一行：

```python
uvicorn.run(app, host="0.0.0.0", port=8100)  # 改为你想要的端口
```

## 许可证

MIT License

## 联系方式

项目地址：/Users/xuhao/work/es/newsoft/fine-tune

---

**Enjoy comparing your models! 🚀**

