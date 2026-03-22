#!/usr/bin/env python3
"""Sen3B 对话脚本"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def interactive(model, tokenizer):
    print("Sen3B 对话模式 - 输入 quit 退出\n")
    while True:
        try:
            user = input("👤 你: ").strip()
            if user.lower() in ['quit', 'exit', 'q']: break
            if not user: continue
            print("🤖 Sen3B:", chat(model, tokenizer, user))
        except KeyboardInterrupt: break
        except Exception as e: print(f"错误: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./model/Qwen1.5-3B")
    parser.add_argument("--prompt")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    print("加载模型...")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, pad_token='<|endoftext|>')
    kw = {"device_map": "auto", "trust_remote_code": True}
    if args.load_in_8bit: kw["quantization_config"] = __import__('transformers').BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit: kw["quantization_config"] = __import__('transformers').BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else: kw["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kw)
    model.eval()
    print("加载完成!\n")
    if args.interactive or not args.prompt: interactive(model, tok)
    else: print("🤖 Sen3B:", chat(model, tok, args.prompt, args.max_new_tokens))

if __name__ == "__main__": main()
