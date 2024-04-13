import argparse
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, AutoModelForSequenceClassification, TrainerCallback
from peft import PeftModel, LoraConfig, PrefixTuningConfig, PromptTuningConfig, PromptEncoderConfig, TaskType, get_peft_model, PromptTuningInit, PromptEncoderReparameterizationType, IA3Config, PeftConfig
import os

def get_Trainer(args):
    try:
        tokenizer = AutoTokenizer.from_pretrained("./bloom-1b1")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
    try:
        model = AutoModelForCausalLM.from_pretrained("./bloom-1b1")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
    split = 'train[:' + str(args.size) + ']'
    ds = load_dataset("yahma/alpaca-cleaned", split=split)
    def process_func(example):
        MAX_LENGTH = 256
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
        response = tokenizer(example["output"] + tokenizer.eos_token)
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
        
        
    if args.method == "bitfit":
        num_param = 0
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False
            else:
                num_param += param.numel()
        all_param = sum(param.numel() for param in model.parameters())
        print(f"trainable params: {num_param} || all params: {all_param} || trainable%: {100*num_param/all_param}")   
    elif args.method == "lora":
        config = LoraConfig(task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif args.method == "prefix_tuning":
        config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10, prefix_projection=True)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif args.method == "prompt_tuning":
        config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text="Below is a conversation between a person and a chatbot.",
                    num_virtual_tokens=len(tokenizer("Below is a conversation between a person and a chatbot.")["input_ids"]),
                    tokenizer_name_or_path="bigscience/bloom-1b1"
                )
        model = get_peft_model(model, config)
    elif args.method == "p_tuning":
        config = PromptEncoderConfig(
                    task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10,
                    encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
                    encoder_dropout=0.1, 
                    #encoder_num_layers=5, 
                    encoder_hidden_size=1024
                )
        model = get_peft_model(model, config)
    elif args.method == "ia3":
        config = IA3Config(task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif args.method == "peft":
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        raise ValueError(f"{args.method} does not exist!")

    
    args = TrainingArguments(
        output_dir="./" + args.method + "/chatbot",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=1
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    return trainer

class LossLoggingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.losses = []
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        losses_path = f"{self.output_dir}/losses.txt"
        if os.path.exists(losses_path):
            open(losses_path, 'w').close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.losses.append(logs['loss'])
            with open(f"{self.output_dir}/losses.txt", "a") as f:
                f.write(f"{state.global_step}: {logs['loss']}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='bitfit')
    parser.add_argument('--size', type=int, default='8000')
    
    args = parser.parse_args()
    
    trainer = get_Trainer(args)
    trainer.add_callback(LossLoggingCallback(output_dir="./"+args.method))
    trainer.train()
    print("train finished.")

if __name__ == '__main__':
    main()
