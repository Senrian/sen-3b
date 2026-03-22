"""Sen3B 模型封装"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional

class Sen3BModel:
    def __init__(self, model_path: str, quantization: Optional[str] = None, device_map: str = "auto"):
        self.model_path = model_path
        self.quantization = quantization
        self.device_map = device_map
        self.tokenizer = None
        self.model = None

    def load(self):
        print(f"加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, pad_token='<|endoftext|>')
        kw = {"device_map": self.device_map, "trust_remote_code": True}
        if self.quantization == "int8": kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "int4": kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        else: kw["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kw)
        self.model.eval()
        print("加载完成!")
        return self

    def chat(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        if not self.model: raise RuntimeError("模型未加载")
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
