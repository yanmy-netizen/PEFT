# Fine-tune of Bloom 1.1B model for Chatbot implementation using PEFT method

## Description
Our project aims to implement PEFT techniques using our personal limited computational resources to fine-tune a conversational chatbot. We explored some of the most commonly used PEFT methods in the field, leveraging the Hugging Face Transformers and PEFT library. We conducted fine-tuning on the base model for 10 epochs using the following methods: BitFit, Prompt tuning, P-tuning, Prefix tuning, LoRa, LoHa, and (IA)$^3$. We then evaluated the fine-tuned models and compared their performance.

## Installation

To install the required dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, you can choose from several methods. Use one of the following commands, replacing `method_name` with one of the supported methods: `bitfit`, `prompt_tuning`, `p_tuning`, `prefix_tuning`, `lora`, `loha`, or `ia3`. For example, if you want to train with `loha`, please run the following command:

```bash
python train.py --method=loha
```

### Post-Training

After training, you can evaluate the model's performance or interact with the trained model using one of the following commands:

- To evaluate the model:
  
  ```bash
  python eval.py --method=loha
  ```

- To chat with the model:

  ```bash
  python chat.py --method=loha
  ```

## Files Description

- `chat.py`: Script for interacting with the trained model.
- `eval.py`: Script to evaluate the model's performance.
- `loss_curve.ipynb`: Notebook for visualizing the training loss curve.
- `requirements.txt`: File containing a list of dependencies to install.
- `train.py`: Script to start the training process.
