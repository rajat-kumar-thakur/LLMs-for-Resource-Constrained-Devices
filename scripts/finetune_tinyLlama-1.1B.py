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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # ~1M parameters

def tokenize_func(examples):
    combined = [p + "\n" + c for p, c in zip(examples["prompt"], examples["code"])]
    return tokenizer(
        combined,
        padding="max_length",
        truncation=True,
        max_length=512,  # Reduced context for efficiency
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_func, 
    batched=True,
    remove_columns=["prompt", "code"]
)

# %%
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

model.gradient_checkpointing_enable()  # Enable before training
model.enable_input_require_grads()  # Ensure parameters require gradients
model.config.use_cache = False  # Required for gradient checkpointing

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8  # Improves performance on modern hardware
)

training_args = TrainingArguments(
    output_dir="./scientific-codegen",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,  # Optimal for LoRA
    optim="paged_adamw_32bit",  # Better for 4-bit training
    logging_steps=20,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    warmup_ratio=0.1,  # Better than fixed steps
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    eval_strategy="no",
    save_total_limit=2,
    gradient_checkpointing=True,
    remove_unused_columns=False,  # Important for PEFT models
    label_names=["input_ids", "attention_mask", "labels"]  # Explicitly specify
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

try:
    print("Starting training...")
    trainer.train()
    
    trainer.model.save_pretrained("./final-model")
    tokenizer.save_pretrained("./final-model")
    print("Training completed successfully!")
    
except RuntimeError as e:
    print(f"Training failed: {str(e)}")
    print("Troubleshooting steps:")
    print("1. Check dataset format - ensure tokenized_dataset has 'input_ids', 'attention_mask', and 'labels'")
    print("2. Reduce batch size or gradient accumulation steps")
    print("3. Try without gradient checkpointing")
    print("4. Verify model supports training (e.g., not quantized too aggressively)")

# %%
import re
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM  # Fixed import

class ScientificCodeGenerator:
    def __init__(self, model_path="./final-model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
    
    def generate(self, task: str, max_length=512):  # Increased max_length
        # Match training prompt format exactly
        prompt = f"Generate Python code for: {task}\nRequirements:\n1. Use sklearn/numpy\n2. Add comments\n\nCode:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Proper generation parameters
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=0.7,  # Higher temperature for creativity
            top_p=0.9,         # Nucleus sampling
            do_sample=True,     # Enable sampling
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2  # Prevent repetition
        )
        
        # Skip special tokens and remove input prompt
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_output.replace(prompt, "").strip()
    
    @staticmethod
    def execute(code: str, data=None):
        """Executes generated code safely"""
        # Extract code block
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        
        # Create restricted environment
        restricted_env = {
            "__builtins__": {},
            "print": print,
            "np": np,
            "pd": pd,
            "plt": plt,
            "KMeans": KMeans,
            "IsolationForest": IsolationForest,
            "RandomForestClassifier": RandomForestClassifier,
            "data": data
        }
        
        try:
            exec(code, restricted_env)
            return restricted_env
        except Exception as e:
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = ScientificCodeGenerator()
    
    # Example 1: Classification
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4)
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(4)])
    data["target"] = y
    
    task = "Classify this data using Random Forest and show accuracy"
    print("Generating code for classification...")
    generated_code = generator.generate(task)
    print("\nGenerated Code:\n", generated_code)
    
    result = generator.execute(generated_code, data)
    if "accuracy" in result:
        print(f"\nClassification Accuracy: {result['accuracy']:.2f}")
    elif "error" in result:
        print(f"\nExecution Error: {result['error']}")
    
    # Example 2: Anomaly Detection
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.5)
    data = pd.DataFrame(X, columns=["x", "y"])
    
    task = "Perform anomaly detection on this 2D data"
    print("\nGenerating code for anomaly detection...")
    generated_code = generator.generate(task)
    print("\nGenerated Code:\n", generated_code)
    result = generator.execute(generated_code, data)
    
    if "anomalies" in result:
        print(f"\nDetected {sum(result['anomalies'])} anomalies")
        plt.scatter(data['x'], data['y'], c=result['anomalies'])
        plt.title("Anomaly Detection Results")
        plt.show()
    elif "error" in result:
        print(f"\nExecution Error: {result['error']}")


