# 更新日志

## v1.1.0 (2025-11-14)

### 新增功能

#### 1. 自动检测模型名称 🎯

现在模型名称字段可以留空，系统会自动从模型路径检测名称！

**示例路径：**
```
/Users/newmind/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct
```

**自动检测结果：**
```
Qwen/Qwen3-Next-80B-A3B-Instruct
```

**检测优先级：**
1. 从 `config.json` 中的 `_name_or_path` 字段读取
2. 从 `config.json` 中的 `model_name` 字段读取
3. 从路径提取最后两级目录（如 `Qwen/Qwen3-Next-80B-A3B-Instruct`）
4. 从路径提取最后一级目录

**使用方法：**
- 只需填写模型路径，名称字段留空即可
- 系统会在加载时自动检测并显示模型名称
- 你也可以手动输入名称来覆盖自动检测的结果

#### 2. 界面全面汉化 🇨🇳

所有界面文本已汉化为中文，更符合中文用户使用习惯。

**汉化内容包括：**
- 标题：模型对比工具
- 副标题：微调模型效果对比平台
- 模型配置区：
  - "模型 A（原始模型）"
  - "模型 B（微调模型）"
  - "模型名称（可选，留空自动检测）"
  - "模型路径"
  - "加载模型" / "卸载" 按钮
- 输入区：
  - "输入提示词"
  - "最大长度" / "温度" / "Top P"
  - "开始对比生成" 按钮
- 输出区：
  - "模型 A 输出" / "模型 B 输出"
  - "加载模型 X 并生成以查看输出"
- 提示信息：
  - "请输入模型路径"
  - "请输入提示词"
  - "请至少加载一个模型"
  - "加载中..." / "生成中..."
  - "错误：XXX"
- 模型信息显示：
  - "设备：cuda"
  - "参数量：7,000,000,000"
  - "精度：torch.float16"

### 技术改进

#### model_manager.py
- 新增 `detect_model_name_from_path()` 函数
- 自动读取 `config.json` 提取模型信息
- 支持 ModelScope、Hugging Face 等常见路径格式
- ModelManager 构造函数的 `model_name` 参数改为可选

#### static/index.html
- 更新所有界面文本为中文
- 优化提示文本的描述性
- placeholder 示例使用实际路径格式
- 错误提示更加友好

### 使用示例

#### 场景 1：使用 ModelScope 下载的模型

```
模型 A 路径：
/Users/newmind/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Instruct

模型 B 路径：
/Users/newmind/.cache/modelscope/hub/models/Qwen/Qwen3-Next-80B-A3B-Finetuned
```

**操作步骤：**
1. 填写模型路径（名称留空）
2. 点击"加载模型"
3. 系统自动检测并显示：`Qwen/Qwen3-Next-80B-A3B-Instruct`
4. 输入问题："请解释什么是人工智能？"
5. 点击"开始对比生成"

#### 场景 2：使用 Hugging Face 下载的模型

```
模型 A 路径：
/Users/newmind/models/llama-2-7b-chat

模型 B 路径：
/Users/newmind/models/llama-2-7b-finetuned
```

自动检测结果：
- 模型 A：`models/llama-2-7b-chat`
- 模型 B：`models/llama-2-7b-finetuned`

### 兼容性

- ✅ 完全向后兼容
- ✅ 仍支持手动输入模型名称
- ✅ 对现有功能无影响

### 已知问题

无

---

## v1.0.0 (2025-11-14)

### 初始发布

- 支持双模型加载和对比
- 流式输出
- 参数可调（Temperature, Top-P, Max Tokens）
- 现代化 Web 界面
- 完整文档

