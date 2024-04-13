import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, AutoModelForSequenceClassification
from peft import PeftModel

def get_model(method):
    if method == "bitfit":
        model = AutoModelForCausalLM.from_pretrained("./bitfit/chatbot/checkpoint-1000")
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
        command = input("Question: ")
        if command == "exit":
            print("Exiting program.")
            break
        ipt = tokenizer("Human: {}\n{}".format(command, "").strip() + "\n\nAssistant: ", return_tensors="pt")
        result = tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
        print(f"Output: {result}")

if __name__ == '__main__':
    main()
