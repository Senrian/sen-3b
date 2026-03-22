#!/usr/bin/env python3
"""Sen3B 基准测试脚本"""
import argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def benchmark_speed(model, tokenizer, num_runs=3):
    print("\n📊 推理速度测试")
    texts = ["用Python写一个快速排序", "介绍一下人工智能"]
    for text in texts:
        messages = [{"role": "user", "content": text}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            before = time.time()
            outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            elapsed = time.time() - before
        print(f"  文本: {text[:20]}... | 耗时: {elapsed:.2f}s | 速度: {100/elapsed:.1f} tok/s")

def benchmark_memory(model):
    if torch.cuda.is_available():
        print(f"\n💾 显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./model/Qwen1.5-3B")
    parser.add_argument("--test", default="all")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    print("加载模型...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    kw = {"device_map": "auto", "trust_remote_code": True}
    if args.load_in_8bit: kw["quantization_config"] = __import__('transformers').BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit: kw["quantization_config"] = __import__('transformers').BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else: kw["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kw)
    model.eval()
    print("模型加载完成!")
    if args.test in ["all", "speed"]: benchmark_speed(model, tok)
    if args.test in ["all", "memory"]: benchmark_memory(model)
    print("\n基准测试完成!")

if __name__ == "__main__": main()
