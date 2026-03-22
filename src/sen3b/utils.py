"""Sen3B 工具函数"""
import torch, time
from typing import Dict

def measure_speed(model, tokenizer, prompt: str, num_runs: int = 3) -> Dict:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    times = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        times.append(time.time() - start)
    return {"avg_time": sum(times)/len(times), "tokens_per_second": 100/(sum(times)/len(times))}

def get_memory() -> Dict:
    if torch.cuda.is_available():
        return {"allocated_gb": torch.cuda.memory_allocated()/1024**3}
    return {}
