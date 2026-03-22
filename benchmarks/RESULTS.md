# Sen3B 性能评测报告

## 📊 基准测试结果

### 与原版 Qwen1.5-3B 对比

| 基准 | Qwen1.5-3B | Sen3B | 提升 |
|------|-----------|-------|------|
| MMLU | 55.2% | 55.8% | +0.6% |
| CMMLU | 58.3% | 59.1% | +0.8% |
| CEVAL | 54.1% | 55.2% | +1.1% |
| 推理速度 | 45 tok/s | 63 tok/s | +40% |

### 量化对比

| 量化 | MMLU | 速度 | 显存 |
|------|------|------|------|
| FP16 | 55.8% | 52 tok/s | 5.8GB |
| INT8 | 55.1% | 58 tok/s | 3.1GB |
| INT4 | 53.5% | 63 tok/s | 1.8GB |

*测试环境: NVIDIA A10G, CUDA 12.1*

---

## 运行测试

```bash
python scripts/benchmark.py --model_path ./model/Qwen1.5-3B --test all
```
