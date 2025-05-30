import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from datasets import load_dataset, load_from_disk
from peft import get_peft_model, LoraConfig, TaskType
from huggingface_hub import login

login(token="")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")

# 1. Loading teacher and student model
# === Load Teacher Model ===
teacher_model_name = "meta-llama/Llama-3.2-3B-Instruct"
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token  # Fix pad token warning

teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.float16
).to(device)
teacher_model.eval()

# === Load Student Model ===
print("loading student model")
student_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_tokenizer.pad_token = student_tokenizer.eos_token
student_model = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)

# === Apply LoRA ===
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    # target_modules=["q_proj", "v_proj"],  # Based on model architecture
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
student_model = get_peft_model(student_model, lora_config)
student_model.print_trainable_parameters()


# 2. Load dataset
# === Load Dataset ===
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")

# 3. Generate teacher responses at first (to save memory during student inference)
# === Generate Teacher Responses ===
print("Generating teacher respones")
def generate_teacher_output(example):
    prompt = example["text"].strip()

    # Always return a dictionary
    if len(prompt) < 10:
        return {"prompt": None, "response": None}

    inputs = teacher_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = teacher_model.generate(**inputs, max_new_tokens=64, pad_token_id=teacher_tokenizer.eos_token_id)
    response = teacher_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return {"prompt": prompt, "response": response}

if os.path.exists("./cached_distilled_data"):
    distilled_data = load_from_disk("./cached_distilled_data")
else:
    distilled_data = raw_dataset.map(generate_teacher_output)
    distilled_data = distilled_data.filter(lambda x: x["response"] is not None and x["prompt"] is not None)
    distilled_data.save_to_disk("./cached_distilled_data")

del teacher_model
torch.cuda.empty_cache()
gc.collect()

# 4. Tokenize student inputs and labels
# === Tokenize for Student ===
print("tokenizing student")
def tokenize_student(example):
    prompt = example["prompt"]
    response = example["response"]
    combined = f"<s>{prompt}</s> {response}"
    enc = student_tokenizer(
        combined,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].squeeze(0)
    attention_mask = enc["attention_mask"].squeeze(0)
    
    # Mask out prompt from loss
    prompt_len = len(student_tokenizer(f"<s>{prompt}</s>")["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = -100  # Ignore loss for prompt tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_dataset = distilled_data.map(tokenize_student, batched=False)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Finetuning student model (under lora)
# === Fine-Tune Student ===
training_args = TrainingArguments(
    output_dir="./distilled_student_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=50,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=student_tokenizer, mlm=False)

trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=student_tokenizer,
    data_collator=data_collator,
)

# === Train ===
trainer.train()

# === Save ===
student_model.save_pretrained("./distilled_student_lora")
student_tokenizer.save_pretrained("./distilled_student_lora")




"""
# testing code
prompt = "What is AI?"
response = "AI stands for Artificial Intelligence."

prompt_tokens = student_tokenizer(f"<s>{prompt}</s>")["input_ids"]
combined_tokens = student_tokenizer(f"<s>{prompt}</s> {response}")["input_ids"]

print("Prompt tokens IDs:", prompt_tokens)
print("Combined tokens IDs:", combined_tokens)

print("Prompt tokens decoded:", student_tokenizer.decode(prompt_tokens))
print("Combined tokens decoded:", student_tokenizer.decode(combined_tokens))

# Check if prompt_tokens are at start of combined_tokens
print("Is prompt tokens prefix of combined tokens?", combined_tokens[:len(prompt_tokens)]
"""