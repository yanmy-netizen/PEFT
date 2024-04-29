# -*- coding: utf-8 -*-
"""evaluate.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Xu_D_Rym2_pHCeX_6Snv0yYTn9oZtI-F
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from peft import PeftModel
import evaluate
from evaluate import load
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='bitfit')
    
args = parser.parse_args()


def get_model(method):
    model_path = f"./{method}/chatbot/checkpoint-10000"
    if method == "bitfit":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif method == "lora":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif method == "loha":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif method == "prefix_tuning":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif method == "prompt_tuning":
        try:
            pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
        except Exception:
            pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
        model = PeftModel.from_pretrained(model=pt_model, model_id=model_path)        
    elif method == "p_tuning":
        try:
            pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
        except Exception:
            pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
        model = PeftModel.from_pretrained(model=pt_model, model_id=model_path)  
    elif method == "ia3":
        try:
            pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
        except Exception:
            pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
        model = PeftModel.from_pretrained(model=pt_model, model_id=model_path)  
    # elif method == "peft":
    #     try:
    #         pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
    #     except Exception:
    #         pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
    #     # lora_config = LoraConfig(
    #     #     r=8,
    #     #     lora_alpha=16,
    #     #     target_modules=["q_proj", "v_proj"],
    #     #     lora_dropout=0.05,
    #     #     bias="none",
    #     #     task_type="CAUSAL_LM"
    #     # )
    #     # model = PeftModel.from_pretrained(model=pt_model, model_id="./peft/checkpoint-1000", lora_config=lora_config) 
    #     model = PeftModel.from_pretrained(model=pt_model, model_id="./peft/checkpoint-1000")     
    else:
        raise ValueError(f"{method} does not exist!")
    return model



"""### Load data"""

#dataset = load_dataset("tatsu-lab/alpaca_eval") this line does not work
#eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
#eval_set.save_to_disk("./AlpacaEval")


eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
eval_set = eval_set.select(range(500))


"""### Load models"""

#tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
base_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
print(f"Load base model successly!")

"""### Evaluation"""

eval_bleu = evaluate.load("bleu")
eval_rouge = evaluate.load("rouge")
eval_bertscore = evaluate.load("bertscore")

def generate_model_output(model, instruction):
    with torch.no_grad():
        input_string = "Human: {}\n{}".format(instruction, "").strip() + "\n\nAssistant: "
        ipt = tokenizer(input_string, return_tensors="pt").to(model.device)
        otpt = model.generate(**ipt, max_new_tokens=256, do_sample=True)[0]
        result = tokenizer.decode(otpt, skip_special_tokens=True).replace(input_string, "")
        return result


base_model.cuda()
base_model.eval()

# pred_base = []

# with torch.no_grad():
#     for data in tqdm(eval_set, desc='Generating base model outputs'):
#         output = generate_model_output(model=base_model, instruction=data['instruction'])
#         pred_base.append(output)

ft_model = get_model(args.method)
print(f"Load model {args.method} successly!")

base_model.cpu()
torch.cuda.empty_cache()
ft_model.cuda()
ft_model.eval()

pred_ft = []
with torch.no_grad():
    for data in tqdm(eval_set, desc=f'Generating model {args.method} outputs'):
        output = generate_model_output(model=ft_model,instruction=data['instruction'])
        pred_ft.append(output)

ref = []
for data in eval_set:
    ref.append(data['output'])

# base_bleu = eval_bleu.compute(predictions = pred_base, references= ref)
# base_rouge = eval_rouge.compute(predictions = pred_base, references= ref)

# base_bert = eval_bertscore.compute(predictions=pred_base, references=ref, lang="en")
# base_bert = base_bert['precision']

# print(base_bleu)
# print(base_rouge)
# print("Average bertscore precision: ")
# print(sum(base_bert)/len(base_bert))

ft_bleu = eval_bleu.compute(predictions = pred_ft, references= ref)
ft_rouge = eval_rouge.compute(predictions = pred_ft, references= ref)

ft_bert = eval_bertscore.compute(predictions=pred_ft, references=ref, lang="en")
ft_bert = ft_bert['precision']

print(ft_bleu)
print(ft_rouge)
print("Average bertscore precision: ")
print(sum(ft_bert)/len(ft_bert))

# Save result
import json
import os

Eval_result = {
    "bleu": ft_bleu,
    "rouge": ft_rouge,
    "bertscore": sum(ft_bert)/len(ft_bert),
}

directory = f"./{args.method}/"
json_filename = "eval_result.json"

if not os.path.exists(directory):
    os.makedirs(directory)

json_file_path = os.path.join(directory, json_filename)

with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(Eval_result, f, ensure_ascii=False, indent=4)