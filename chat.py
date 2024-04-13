import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig

def get_model(method):
    if method == "bitfit":
        model = AutoModelForCausalLM.from_pretrained("./bitfit/chatbot/checkpoint-1000")
    elif method == "lora":
        model = AutoModelForCausalLM.from_pretrained("./lora/chatbot/checkpoint-1000")
    elif method == "prefix_tuning":
        model = AutoModelForCausalLM.from_pretrained("./prefix_tuning/chatbot/checkpoint-1000")
    elif method == "prompt_tuning":
        try:
            pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
        except Exception:
            pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
        model = PeftModel.from_pretrained(model=pt_model, model_id="./prompt_tuning/chatbot/checkpoint-1000")        
    elif method == "p_tuning":
        try:
            pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
        except Exception:
            pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
        model = PeftModel.from_pretrained(model=pt_model, model_id="./p_tuning/chatbot/checkpoint-1000")  
    elif method == "ia3":
        try:
            pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
        except Exception:
            pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
        model = PeftModel.from_pretrained(model=pt_model, model_id="./ia3/chatbot/checkpoint-1000")  
    elif method == "peft":
        try:
            pt_model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
        except Exception:
            pt_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
        # lora_config = LoraConfig(
        #     r=8,
        #     lora_alpha=16,
        #     target_modules=["q_proj", "v_proj"],
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        # model = PeftModel.from_pretrained(model=pt_model, model_id="./peft/checkpoint-1000", lora_config=lora_config) 
        model = PeftModel.from_pretrained(model=pt_model, model_id="./peft/checkpoint-1000")     
    else:
        raise ValueError(f"{method} does not exist!")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='bitfit')
    
    args = parser.parse_args()
    
    print(f"Welcome! You have chosen the '{args.method}' method to chat. Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./bloom-1b1")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
    model = get_model(args.method)
    print("Load model successly! How can I help you today? Type 'exit' to quit.")
    
    while True:
        command = input("Human: ")
        if command == "exit":
            print("Exiting program.")
            break
        ipt = tokenizer("Human: {}\n{}".format(command, "").strip() + "\n\nAssistant: ", return_tensors="pt")
        result = tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True).replace("Human: " + command, "")
        print(f"{result}")

if __name__ == '__main__':
    main()
