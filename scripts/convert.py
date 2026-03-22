#!/usr/bin/env python3
"""Sen3B 模型转换脚本"""
import argparse, torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def convert(model_path, output, quantize):
    print(f"转换: {model_path} -> {output} ({quantize})")
    kw = {"trust_remote_code": True}
    if quantize == "int8": kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "int4": kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else: kw["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, **kw)
    model.save_pretrained(output)
    print("完成!")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--quantize", default="fp16", choices=["fp16","int8","int4"])
    convert(**vars(p.parse_args()))
