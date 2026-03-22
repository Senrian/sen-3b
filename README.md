# Sen3B - 轻量级大语言模型

<p align="center">
  <img src="https://img.shields.io/badge/Parameters-3B-blue.svg" alt="Parameters">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Base%20Model-Qwen1.5--3B-orange.svg" alt="Base Model">
</p>

<p align="center">
  <strong>Sen3B</strong> — 基于 Qwen1.5-3B 优化的小型语言模型，专为高效推理和边缘部署而设计。
</p>

<p align="center">
  <a href="https://github.com/Senrian/sen-3b">🏠 GitHub</a>
  •
  <a href="https://huggingface.co/Qwen/Qwen1.5-3B">🤗 Hugging Face</a>
  •
  <a href="docs/zh/README.md">📖 中文文档</a>
</p>

---

## 📌 项目简介

Sen3B 是基于 [Qwen1.5-3B](https://huggingface.co/Qwen/Qwen1.5-3B) 优化的开源小模型。

### 主要优化

- 🚀 **推理加速** — 推理速度提升 40%
- 📦 **模型压缩** — INT4/INT8 量化支持
- 🔧 **高效微调** — LoRA 开箱即用
- 📚 **中文增强** — 中文理解能力优化

## 📥 模型下载

请参考 [下载指南](docs/zh/DOWNLOAD.md)

## 🚀 快速开始

```bash
# 安装依赖
pip install torch transformers accelerate bitsandbytes peft

# 运行对话
python scripts/chat.py --model_path ./model/Qwen1.5-3B --interactive
```

## 📊 性能对比

| 模型 | MMLU | CMMLU | 推理速度 |
|------|------|-------|----------|
| Qwen1.5-3B | 55.2% | 58.3% | 45 tok/s |
| **Sen3B** | **55.8%** | **59.1%** | **63 tok/s** |

## 📁 项目结构

```
sen-3b/
├── scripts/           # 运行脚本
├── configs/          # 配置文件
├── src/sen3b/        # 源代码
├── docs/             # 文档
├── benchmarks/        # 基准测试
└── model/            # 模型文件
```

## 📚 文档

- [中文文档](docs/zh/README.md)
- [快速开始](docs/zh/QUICKSTART.md)
- [下载指南](docs/zh/DOWNLOAD.md)
- [性能报告](benchmarks/RESULTS.md)

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

MIT License
