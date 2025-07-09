# %%

# %%
import json
import os
import subprocess
from datasets import Dataset, load_dataset
from typing import List

def prepare_datasets() -> Dataset:
    """Prepare dataset using only real data sources"""
    all_prompts: List[str] = []
    all_codes: List[str] = []
    
    def load_scidocs() -> List[str]:
        try:
            os.makedirs("scidocs_data", exist_ok=True)
            if not os.path.exists("scidocs_data/paper_metadata_view_cite_read.json"):
                subprocess.run([
                    "aws", "s3", "sync", "--no-sign-request",
                    "s3://ai2-s2-research-public/specter/scidocs/",
                    "scidocs_data/", "--region", "us-west-2", "--quiet"
                ], check=True)
            
            with open("scidocs_data/paper_metadata_view_cite_read.json", "r") as f:
                data = json.load(f)
            
            prompts = []
            for paper_id, content in data.items():
                title = content.get('title', '') or ''
                abstract = content.get('abstract', '') or ''
                
                if len(title) > 10 and len(abstract) > 200:
                    prompts.append(
                        f"Generate Python code for: {title}\nAbstract: {abstract[:400]}"
                    )
            return prompts
        except Exception as e:
            print(f"SciDocs loading failed: {str(e)}")
            return []

    def load_astronomy() -> List[str]:
        try:
            ds = load_dataset("David-Xu/astronomy-stack-dpo-text", split="train")
            return [example['prompt'] for example in ds]
        except Exception as e:
            print(f"Astronomy dataset loading failed: {str(e)}")
            return []

    def load_science() -> List[str]:
        try:
            ds = load_dataset("millawell/wikipedia_field_of_science", split="train")
            return [text for text in ds['text'] if len(text) > 30]
        except Exception as e:
            print(f"Science dataset loading failed: {str(e)}")
            return []

    def load_code_samples() -> List[str]:
        try:
            ds = load_dataset("bigcode/the-stack", 
                            data_dir="data/python", 
                            split="train",
                            streaming=True)
            
            samples = []
            for sample in ds:
                content = sample["content"]
                if any(imp in content for imp in ["numpy", "sklearn", "pandas", "matplotlib"]):
                    if "auto-generated" not in content.lower():
                        samples.append(content[:2000])
                        if len(samples) >= 20000:
                            break
            return samples
        except Exception as e:
            print(f"Code dataset loading failed: {str(e)}")
            return []

    # Load all datasets
    scidocs = load_scidocs()[:25000]  # Cap at 25k
    astronomy = load_astronomy()[:15000]  # Cap at 15k
    science = load_science()[:15000]  # Cap at 15k
    code_samples = load_code_samples()[:20000]  # Cap at 20k
    
    science_code_prompts = [
        f"Generate Python code for: {text.split(':')[-1].strip()}" 
        for text in science[:10000]
    ]

    # Combine all sources
    all_prompts.extend(scidocs)
    all_prompts.extend(astronomy)
    all_prompts.extend(science)
    all_prompts.extend(science_code_prompts)
    all_prompts.extend(["Generate Python code:"] * len(code_samples))
    
    all_codes.extend([""] * (len(scidocs) + len(astronomy) + len(science) + len(science_code_prompts)))
    all_codes.extend(code_samples)

    # Final dataset
    return Dataset.from_dict({
        "prompt": all_prompts,
        "code": all_codes
    })

dataset = prepare_datasets()
print(f"Final dataset size: {len(dataset)}")
print("Sample prompts:", dataset["prompt"][:3])
print("Sample codes:", [c[:100] + "..." if c else "" for c in dataset["code"][-3:]])

# %%
import random

indices = random.sample(range(len(dataset)), 5)

for i in indices:
    sample = dataset[i]
    print(f"\nSample {i + 1}")
    print("Prompt:", sample["prompt"])
    print("Code:", sample["code"])

# %%
print(f"Model device: {next(model.parameters()).device}")

# %%
for name, param in model.named_parameters():
    if param.device != device:
        print(f"Parameter {name} on wrong device: {param.device}")

# %%
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "bigcode/tiny_starcoder_py"

device = "cuda" if torch.cuda.is_available() else "cpu"
device_index = torch.cuda.current_device() if device == "cuda" else None
print(f"Using device: {device} (index: {device_index})")

device_map = {"": device_index if device == "cuda" else device}
print(f"Device map: {device_map}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,  # Use fixed device mapping
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

def tokenize_func(examples):
    combined = [f"# {p}\n{c}" for p, c in zip(examples["prompt"], examples["code"])]
    return tokenizer(
        combined,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_func, 
    batched=True,
    remove_columns=["prompt", "code"]
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir="./scientific-codegen",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    optim="paged_adamw_8bit",
    logging_steps=20,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    eval_strategy="no",
    save_total_limit=2,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    max_grad_norm=0.3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

model.save_pretrained("./codegen-lora-adapters")
tokenizer.save_pretrained("./codegen-lora-adapters")
print("Training complete! Adapters saved.")

# %%
from peft import PeftModel
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "bigcode/tiny_starcoder_py"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"
device_index = torch.cuda.current_device() if device == "cuda" else None
print(f"Using device: {device} (index: {device_index})")

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./codegen-lora-adapters")
model = model.merge_and_unload()

prompt = "# Write a Python function to calculate factorial\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.2,
    pad_token_id=tokenizer.eos_token_id
)
print(tokenizer.decode(outputs[0]))

# %%



