# Quick Start Guide - 快速开始

## 5 分钟快速上手

### Step 1: 安装依赖（1 分钟）

```bash
cd /Users/xuhao/work/es/newsoft/fine-tune

# 使用 uv 自动安装
./start.sh
```

如果 `start.sh` 没有执行权限：

```bash
chmod +x start.sh
./start.sh
```

或手动安装：

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -e .

# 启动服务
python main.py
```

### Step 2: 打开界面（10 秒）

浏览器访问：**http://localhost:8100**

### Step 3: 加载模型（2-5 分钟）

#### Model A (原始模型)
```
Model Name: Original Llama 2
Model Path: /Users/xuhao/models/llama-2-7b-chat
```

点击 **Load Model** → 等待加载完成（状态指示器变绿）

#### Model B (Fine-tuned 模型)
```
Model Name: Fine-tuned Llama 2
Model Path: /Users/xuhao/models/llama-2-7b-finetuned
```

点击 **Load Model** → 等待加载完成

### Step 4: 测试对比（30 秒）

在 **Input Prompt** 输入：

```
请解释一下什么是机器学习？
```

点击 **🚀 Generate Comparison**

观察左右两列的输出差异！

---

## 完整示例

### 示例 1: 本地 Hugging Face 模型

假设你已经下载了模型到本地：

```python
# 下载模型（只需一次）
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
save_path = "/Users/xuhao/models/llama-2-7b"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

然后在界面填写 `/Users/xuhao/models/llama-2-7b`

### 示例 2: Fine-tuned 模型

如果你已经 fine-tune 了模型：

```python
# Fine-tuning 后保存
fine_tuned_model.save_pretrained("/Users/xuhao/models/my-finetuned-model")
tokenizer.save_pretrained("/Users/xuhao/models/my-finetuned-model")
```

在界面填写 `/Users/xuhao/models/my-finetuned-model`

---

## 测试问题建议

### 通用测试

```
1. 请解释什么是深度学习？
2. 写一个 Python 函数来计算斐波那契数列
3. 用三句话总结二战的历史影响
```

### 领域特定测试（根据你的 fine-tune 目标）

```
医疗领域:
- 请解释高血压的症状和治疗方法

法律领域:
- 解释合同违约的法律责任

技术领域:
- 如何优化 React 应用的性能？
```

---

## 常见问题速查

### ❌ 模型加载失败

```bash
# 检查路径
ls /path/to/model

# 应该包含这些文件:
# - config.json
# - pytorch_model.bin (或 model.safetensors)
# - tokenizer.json
# - tokenizer_config.json
```

### ❌ 内存不足

```bash
# 方案 1: 只加载一个模型
# 方案 2: 使用较小的模型（3B/7B 而非 13B）
# 方案 3: 使用量化版本
```

### ❌ 生成速度慢

```bash
# CPU 模式很慢是正常的
# 建议使用 GPU（NVIDIA + CUDA）

# 安装 CUDA 版 PyTorch:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ 端口被占用

```bash
# 修改 main.py 最后一行:
uvicorn.run(app, host="0.0.0.0", port=8101)  # 改为其他端口
```

---

## 性能参考

### CPU 模式
- 7B 模型: ~1-3 tokens/s
- 加载时间: ~30-60s

### GPU 模式 (NVIDIA RTX 3090)
- 7B 模型: ~20-40 tokens/s
- 加载时间: ~10-20s

---

## 下一步

1. 查看完整文档: [README.md](README.md)
2. 查看设计文档: `/Users/xuhao/work/es/newsoft/docmanage/20251114_fine_tune_comparison_tool.md`
3. API 文档: http://localhost:8100/docs

---

**祝你使用愉快！🎉**

有问题？检查服务日志输出的错误信息。

