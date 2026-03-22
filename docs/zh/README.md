# Sen3B 中文文档

## 项目简介

Sen3B 是基于 Qwen1.5-3B 优化的开源小模型，专为高效推理和边缘部署设计。

### 与原版对比

| 特性 | Qwen1.5-3B | Sen3B |
|------|------------|-------|
| 参数规模 | 3B | 3B |
| 推理速度 | 45 tok/s | 63 tok/s |
| 中文能力 | 一般 | 优化增强 |
| 量化支持 | FP16 | INT4/INT8/FP16 |

---

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+
- 16GB+ RAM
- 8GB+ GPU 显存

### 安装依赖

```bash
pip install torch transformers accelerate
pip install bitsandbytes peft
```

### 下载模型

```bash
huggingface-cli download Qwen/Qwen1.5-3B --local-dir ./model/Qwen1.5-3B
```

### 运行对话

```python
from src.sen3b import Sen3BModel

model = Sen3BModel("./model/Qwen1.5-3B")
model.load()
response = model.chat("你好，请介绍一下人工智能")
print(response)
```

---

## 量化部署

### INT8 量化

```python
model = Sen3BModel("./model/Qwen1.5-3B", quantization="int8")
model.load()
```

### INT4 量化

```python
model = Sen3BModel("./model/Qwen1.5-3B", quantization="int4")
model.load()
```

---

## 微调训练

### LoRA 配置

```json
{
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}
```

### 开始微调

```bash
python scripts/train.py --config configs/lora_config.json
```

---

## 常见问题

**Q: 显存不足怎么办？**
A: 使用 INT8 或 INT4 量化版本。

**Q: 如何提升推理速度？**
A: 使用量化、启用 KV Cache、使用批处理。

---

## 联系方式

GitHub: https://github.com/Senrian/sen-3b
