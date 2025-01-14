from peft import PeftModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def merge(model, lora_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model)
    lm_model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)

    lora_model = PeftModel.from_pretrained(
        lm_model,
        lora_path,
    )
    lm_model = lora_model.merge_and_unload()
    return lm_model, tokenizer

lm_model, tokenizer = merge("/home/yg/PycharmProjects/Flab-Pruner-local/save_models/CodeQwen1.5-base", 
                            "./save_models/CodeQwen-instruct")
tokenizer.save_pretrained("./save_models/CQ_instruct")
lm_model.save_pretrained("./save_models/CQ_instruct")