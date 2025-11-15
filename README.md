# Model Comparison Tool

Fine-tuned Model Side-by-Side Comparison Tool - 模型对比工具

## 功能特性

- ✅ 支持加载两个本地 MLX 格式的模型（Apple Silicon 优化）
- ✅ 支持 LoRA 适配器加载和对比
- ✅ 自动查找最新的适配器文件
- ✅ 支持基础模型和 Fine-tuned 模型对比
- ✅ 流式输出，实时对比两个模型的响应
- ✅ 双列并排显示，方便观察差异
- ✅ 支持自定义生成参数（Temperature, Top-P, Max Tokens）
- ✅ 响应时间统计
- ✅ 现代化的 Web 界面

## 系统要求

- Python 3.11+
- uv (Python 包管理器)
- **Apple Silicon (M1/M2/M3)** - MLX 框架需要 Apple Silicon Mac
- mlx-lm (MLX 语言模型库)

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

# 安装 MLX-LM（如果还没有安装）
uv pip install mlx-lm
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

#### 方式 1: 使用 API 加载（推荐）

通过 API 加载模型，支持以下配置：

**加载基础模型（不使用适配器）**：
```json
POST /api/models/model_a/load
{
  "model_path": "/Users/newmind/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
  "model_name": "Qwen3-Coder-30B (Base)",
  "no_adapter": true
}
```

**加载带适配器的模型（自动查找最新的）**：
```json
POST /api/models/model_b/load
{
  "model_path": "/Users/newmind/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
  "model_name": "Qwen3-Coder-30B (Fine-tuned)",
  "adapter_path": null,  // null 表示自动查找最新的适配器
  "saves_dir": "mlx/saves/qwen-lora"
}
```

**加载指定适配器的模型**：
```json
POST /api/models/model_b/load
{
  "model_path": "/Users/newmind/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
  "model_name": "Qwen3-Coder-30B (Fine-tuned)",
  "adapter_path": "mlx/saves/qwen-lora/train_2025-11-15-00-36-05/adapters.npz"
}
```

#### 方式 2: 通过 Web 界面加载

1. 在 **Model A** 区域填写：
   - **Model Name**: 原始模型名称（可选）
   - **Model Path**: 模型的本地路径或 HuggingFace 模型 ID
   - **No Adapter**: 勾选此选项使用基础模型（不加载适配器）
   - 点击 **Load Model** 加载模型

2. 在 **Model B** 区域填写：
   - **Model Name**: Fine-tuned 模型名称（可选）
   - **Model Path**: 相同的模型路径
   - **Adapter Path**: 适配器文件路径（留空则自动查找最新的）
   - 点击 **Load Model** 加载模型

> **提示**：
> - 模型路径可以是本地路径或 HuggingFace 模型 ID（如 `mlx-community/Qwen2.5-3B-Instruct-4bit`）
> - 适配器路径示例：`mlx/saves/*/train_2025-11-15-00-36-05/adapters.npz`
> - 如果不指定适配器路径，系统会自动查找 `mlx/saves/*` 目录下最新的适配器

### 步骤 2: 输入问题

在 **Input Prompt** 区域输入你想问的问题，例如：

```
请解释一下机器学习中的过拟合问题。
```

### 步骤 3: 调整参数（可选）

- **Max Tokens**: 生成的最大 token 数量（默认 512）
- **Top P**: Nucleus sampling 参数（默认 0.9）

### 步骤 4: 生成对比

点击 **🚀 Generate Comparison** 按钮，两个模型将同时开始生成响应。

你可以实时看到：
- 左侧：Model A 的响应
- 右侧：Model B 的响应
- 每个模型的响应时间

## 模型和适配器

### 支持的模型格式

- **MLX 格式模型**：支持本地 MLX 模型路径或 HuggingFace 模型 ID
- **LoRA 适配器**：支持 MLX-LM 训练的 LoRA 适配器（`.npz` 文件）

### 模型路径示例

1. **本地 MLX 模型路径**：
   ```
   /Users/newmind/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit
   ```

2. **HuggingFace 模型 ID**（自动下载）：
   ```
   mlx-community/Qwen2.5-3B-Instruct-4bit
   ```

### 适配器路径

适配器文件通常保存在 `mlx/saves/qwen-lora/train_YYYY-MM-DD-HH-MM-SS/adapters.npz`

**自动查找适配器**：
- 如果不指定 `adapter_path`，系统会自动在 `mlx/saves/qwen-lora` 目录下查找最新的适配器
- 查找逻辑：按修改时间排序，选择最新的 `train_*/adapters.npz` 文件

**手动指定适配器**：
- 提供完整的适配器文件路径，例如：
  ```
  mlx/saves/qwen-lora/train_2025-11-15-00-36-05/adapters.npz
  ```

### 使用场景

1. **对比基础模型 vs Fine-tuned 模型**：
   - Model A: 基础模型（`no_adapter: true`）
   - Model B: 带适配器的模型（指定 `adapter_path`）

2. **对比不同版本的 Fine-tuned 模型**：
   - Model A: 使用适配器 A
   - Model B: 使用适配器 B

## API 文档

启动服务后，访问 http://localhost:8100/docs 查看完整的 API 文档（Swagger UI）。

### 主要 API 端点

- `POST /api/models/{model_id}/load` - 加载模型
  - 请求体：`ModelConfig`（包含 `model_path`, `adapter_path`, `no_adapter` 等）
- `POST /api/models/{model_id}/unload` - 卸载模型
- `GET /api/models/{model_id}/status` - 获取模型状态
- `POST /api/generate/stream` - 流式生成文本
- `GET /api/status` - 获取所有模型状态

### API 使用示例

**Python 示例**：
```python
import requests

# 加载基础模型
response = requests.post("http://localhost:8100/api/models/model_a/load", json={
    "model_path": "/Users/newmind/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "no_adapter": True
})

# 加载带适配器的模型
response = requests.post("http://localhost:8100/api/models/model_b/load", json={
    "model_path": "/Users/newmind/.lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit",
    "adapter_path": "mlx/saves/qwen-lora/train_2025-11-15-00-36-05/adapters.npz"
})

# 生成文本
response = requests.post("http://localhost:8100/api/generate/stream", json={
    "prompt": "什么是 Elasticsearch？",
    "model_id": "model_a",
    "max_new_tokens": 500,
    "temperature": 0.7
})
```

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

**错误**: `Model path does not exist` 或 `MLX-LM 未安装`

**解决**:
- 确认已安装 MLX-LM: `pip install mlx-lm`
- 确认模型路径正确（本地路径或 HuggingFace 模型 ID）
- 确认在 Apple Silicon Mac 上运行（MLX 需要 Apple Silicon）
- 检查模型目录是否包含必要的文件（`config.json` 等）

### 问题 1.1: 适配器未找到

**错误**: `适配器文件不存在` 或 `未找到任何适配器文件`

**解决**:
- 确认适配器文件路径正确
- 检查 `mlx/saves/qwen-lora` 目录是否存在
- 确认适配器文件格式为 `.npz`
- 如果不指定路径，系统会自动查找最新的适配器

### 问题 2: 内存不足

**错误**: Python 内存错误或系统内存不足

**解决**:
- 使用较小的模型（如 3B 或 7B 模型）
- 使用 4-bit 量化模型（推荐）
- 一次只加载一个模型
- 减少 `max_new_tokens` 参数
- 关闭其他占用内存的应用

### 问题 3: 生成速度慢

**解决**:
- 确保在 Apple Silicon Mac 上运行（MLX 需要 Apple Silicon）
- 使用较小的模型（3B 或 7B）
- 使用 4-bit 量化模型
- 减少 `max_new_tokens`
- 检查其他程序是否占用资源
- 确保使用 MLX 而非 PyTorch（MLX 在 Apple Silicon 上更快）

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
- **Model Loading**: MLX-LM (Apple Silicon 优化)
- **Framework**: MLX (Apple 的机器学习框架)
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

