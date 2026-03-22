# Sen3B 快速开始

## 5分钟上手

### 1. 克隆项目

```bash
git clone https://github.com/Senrian/sen-3b.git
cd sen-3b
```

### 2. 下载模型

```bash
huggingface-cli download Qwen/Qwen1.5-3B --local-dir ./model/Qwen1.5-3B
```

### 3. 安装依赖

```bash
pip install torch transformers accelerate bitsandbytes peft
```

### 4. 运行对话

```bash
python scripts/chat.py --model_path ./model/Qwen1.5-3B --interactive
```

---

## 环境要求

| 要求 | 最低 | 推荐 |
|------|------|------|
| Python | 3.10 | 3.11 |
| 显存 | 4GB (INT8) | 8GB+ |
| 内存 | 8GB | 16GB |

---

## 下一步

- 阅读[完整文档](README.md)
- 查看[性能报告](../benchmarks/RESULTS.md)
- 学习[微调配置](../configs/lora_config.json)
