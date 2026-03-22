# Sen3B 下载指南

## ⚠️ 重要说明

模型文件较大（约 6GB），请确保网络稳定。

---

## 方式一：Hugging Face CLI（推荐）

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download Qwen/Qwen1.5-3B --local-dir ./model/Qwen1.5-3B
```

---

## 方式二：ModelScope 镜像

```bash
pip install modelscope

python << 'EOF'
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen1.5-3B', cache_dir='./model')
EOF
```

---

## 方式三：国内镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen1.5-3B --local-dir ./model/Qwen1.5-3B
```

---

## 方式四：Git LFS

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen1.5-3B ./model/Qwen1.5-3B
```

---

## 验证下载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("./model/Qwen1.5-3B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./model/Qwen1.5-3B", device_map="auto", trust_remote_code=True)
print("模型加载成功!")
```
