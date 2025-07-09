# %%
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import ast
import json
import os
import subprocess
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_blobs
from datasets import load_dataset, Dataset
import torch.nn.utils.prune as prune
import sklearn
import datetime
import gc
from peft import PeftModel

CPU_MODE = False
TEST_SAMPLE_SIZE = 50
OPTIMIZATION_TECHNIQUES = [
    "Base Model",
    "Pruning",
    "Weight Sharing",
    "Early Exit"
]

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"optimization_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"All results will be saved in: {results_dir}")

base_model_name = "bigcode/tiny_starcoder_py"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=None
)
model = PeftModel.from_pretrained(base_model, "./codegen-lora-adapters")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./final-model")
tokenizer.save_pretrained("./final-model")

def apply_optimization(technique_name):
    model = AutoModelForCausalLM.from_pretrained("./final-model", device_map=None)
    if technique_name == "Pruning":
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "lora" not in name.lower():
                prune.ln_structured(module, name='weight', amount=0.15, n=2, dim=0)
                prune.remove(module, 'weight')
        return model
    if technique_name == "Weight Sharing":
        if hasattr(model, 'lm_head') and hasattr(model, 'model'):
            if hasattr(model.model, 'embed_tokens'):
                model.lm_head.weight = model.model.embed_tokens.weight
        return model
    return model

def load_test_datasets():
    test_prompts = []
    try:
        scidocs_path = "scidocs_data"
        os.makedirs(scidocs_path, exist_ok=True)
        if not os.path.exists(os.path.join(scidocs_path, "paper_metadata_view_cite_read.json")):
            subprocess.run([
                "aws", "s3", "sync", "--no-sign-request",
                "s3://ai2-s2-research-public/specter/scidocs/",
                scidocs_path, "--region", "us-west-2", "--quiet"
            ], check=True)
        with open(os.path.join(scidocs_path, "paper_metadata_view_cite_read.json"), "r") as f:
            scidocs_data = json.load(f)
        for i, (paper_id, content) in enumerate(scidocs_data.items()):
            if i >= 10: break
            title = content.get('title', '') or ''
            abstract = content.get('abstract', '') or ''
            if len(title) > 10 and len(abstract) > 200:
                test_prompts.append({
                    "text": (
                        f"Generate complete Python code for: {title}\n"
                        f"Abstract: {abstract[:300]}\n"
                        "Create a synthetic dataset and implement analysis."
                    ),
                    "source": "scidocs",
                    "type": "scientific"
                })
    except Exception as e:
        print(f"SciDocs loading failed: {str(e)}")
    try:
        astronomy = load_dataset("David-Xu/astronomy-stack-dpo-text", split="train")
        for i, example in enumerate(astronomy):
            if i >= 10: break
            test_prompts.append({
                "text": (
                    f"Generate Python code to solve: {example['prompt']}\n"
                    "Create any necessary synthetic data."
                ),
                "source": "astronomy",
                "type": "problem_solving"
            })
    except Exception as e:
        print(f"Astronomy dataset loading failed: {str(e)}")
    try:
        science = load_dataset("millawell/wikipedia_field_of_science", split="train")
        for i, example in enumerate(science):
            if i >= 10: break
            test_prompts.append({
                "text": (
                    f"Generate classification code for: {example['text']}\n"
                    "Create a synthetic dataset and implement classification."
                ),
                "source": "wikipedia_science",
                "type": "classification"
            })
    except Exception as e:
        print(f"Science dataset loading failed: {str(e)}")
    for i in range(10):
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=i)
        data = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(4)])
        data["target"] = y
        test_prompts.append({
            "text": (
                "Create a RandomForest classifier with train-test split and show accuracy\n"
                "The input data is in a DataFrame 'df' with features and 'target' column\n"
                "Steps:\n"
                "1. Split into features (X) and target (y)\n"
                "2. Create train/test splits\n"
                "3. Train classifier\n"
                "4. Make predictions\n"
                "5. Print accuracy"
            ),
            "data": data,
            "source": "synthetic",
            "type": "classification"
        })
        X, y = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=i)
        data = pd.DataFrame(X, columns=["x", "y"])
        test_prompts.append({
            "text": (
                "Perform K-means clustering and visualize results\n"
                "The input data is in a DataFrame 'df' with columns 'x' and 'y'\n"
                "Steps:\n"
                "1. Prepare data\n"
                "2. Fit KMeans model\n"
                "3. Predict clusters\n"
                "4. Create scatter plot colored by cluster\n"
                "5. Save plot as 'clusters.png'"
            ),
            "data": data,
            "source": "synthetic",
            "type": "clustering"
        })
    print(f"Created {len(test_prompts)} test prompts")
    return test_prompts

def generate_robust_code(generator, prompt_text, task_type):
    task_instructions = {
        "scientific": (
            "Implement complete scientific analysis using numpy/pandas\n"
            "Requirements:\n"
            "1. Create synthetic dataset\n"
            "2. Perform meaningful calculations\n"
            "3. Print clear results\n"
            "4. DO NOT just import libraries without using them"
        ),
        "problem_solving": (
            "Solve with scientific computing techniques\n"
            "Requirements:\n"
            "1. Define the problem\n"
            "2. Implement complete solution\n"
            "3. Print the final answer\n"
            "4. DO NOT just import libraries without using them"
        ),
        "classification": (
            "Use RandomForestClassifier with train-test split\n"
            "Requirements:\n"
            "1. Split data into features (X) and target (y)\n"
            "2. Create train/test splits\n"
            "3. Train classifier\n"
            "4. Make predictions\n"
            "5. Print accuracy\n"
            "6. DO NOT just import libraries without using them"
        ),
        "clustering": (
            "Use KMeans clustering and visualize results\n"
            "Requirements:\n"
            "1. Prepare data\n"
            "2. Fit KMeans model\n"
            "3. Predict clusters\n"
            "4. Create scatter plot colored by cluster\n"
            "5. Save plot as 'clusters.png'\n"
            "6. DO NOT just import libraries without using them"
        )
    }.get(task_type, "Implement complete solution with meaningful operations")
    
    structured_prompt = f"""
Generate complete, self-contained Python code to solve:
{prompt_text}

Specific Instructions:
{task_instructions}

Code must:
- Use ONLY numpy, pandas, sklearn, matplotlib
- Print results clearly
- For visualizations: plt.savefig('output.png')
- Have NO unused imports
- Be syntactically correct

Code:
```python
"""
    try:
        output = generator(
            structured_prompt,
            temperature=0.1,
            max_new_tokens=512,
            truncation=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95
        )
        return output[0]['generated_text']
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return ""

def validate_code(generated_code):
    if not generated_code:
        return "", {"numpy": False, "pandas": False, "sklearn": False, "matplotlib": False}
    
    if "```python" in generated_code:
        code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        code = generated_code.split("```")[1].split("```")[0]
    else:
        code = generated_code
    
    lib_usage = {
        "numpy": False,
        "pandas": False,
        "sklearn": False,
        "matplotlib": False
    }
    
    repairs = [
        (r"from\s+sklearn\s+import\s+\*", 
         "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.cluster import KMeans"),
        (r"classifier\.fit\(\)", "classifier.fit(X_train, y_train)"),
        (r"predict\(\)", "predict(X_test)"),
        (r"plt\.show\(\)", "plt.savefig('output.png')"),
        (r"import matplotlib\.pyplot as plt", 
         "import matplotlib.pyplot as plt\nplt.switch_backend('Agg')"),
        (r"\.to_csv\('data\.csv'\)", "")
    ]
    
    for pattern, replacement in repairs:
        code = re.sub(pattern, replacement, code)
    
    if "np." in code or "numpy." in code:
        lib_usage["numpy"] = True
    if "pd." in code or "pandas." in code:
        lib_usage["pandas"] = True
    if "train_test_split" in code or "RandomForestClassifier" in code or "KMeans" in code:
        lib_usage["sklearn"] = True
    if "plt." in code or "matplotlib." in code:
        lib_usage["matplotlib"] = True
    
    required_imports = []
    if lib_usage["numpy"]:
        required_imports.append("import numpy as np")
    if lib_usage["pandas"]:
        required_imports.append("import pandas as pd")
    if lib_usage["matplotlib"]:
        required_imports.append("import matplotlib.pyplot as plt")
    if lib_usage["sklearn"]:
        required_imports.append("from sklearn.ensemble import RandomForestClassifier")
        required_imports.append("from sklearn.cluster import KMeans")
        required_imports.append("from sklearn.model_selection import train_test_split")
        required_imports.append("from sklearn.metrics import accuracy_score")
    
    for imp in required_imports:
        if imp not in code:
            code = imp + "\n" + code
    
    code_lines = code.split('\n')
    cleaned_lines = []
    for line in code_lines:
        if line.strip().startswith("import") or line.strip().startswith("from"):
            lib_name = line.split()[1].split(".")[0] if "import" in line else line.split()[1]
            if lib_name in ["numpy", "np", "pandas", "pd", "sklearn", "matplotlib", "plt"]:
                if any(f"{lib_name}." in line for line in code_lines):
                    cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines), lib_usage

def safe_execute(code: str, data=None):
    if not code:
        return {"status": "error", "message": "Empty code"}
    
    safe_env = {
        "__builtins__": {
            'print': print, 'range': range, 'len': len, 'str': str, 'int': int, 
            'float': float, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 
            'set': set, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'enumerate': enumerate, 'zip': zip
        },
        "np": np,
        "pd": pd,
        "plt": plt,
        "RandomForestClassifier": RandomForestClassifier,
        "KMeans": KMeans,
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
    }
    
    if data is not None:
        safe_env["df"] = data
        
    try:
        ast.parse(code)
        exec(code, safe_env)
        return {"status": "success", "env": safe_env}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {str(e)}"}

def measure_inference_performance(generator, prompt_text, num_runs=5):
    metrics = {
        "avg_latency": 0,
        "throughput": 0,
        "memory_usage": 0,
        "success_rate": 0
    }
    successes = 0
    latencies = []
    try:
        _ = generator(prompt_text, max_new_tokens=50)
        start_time = time.time()
        for _ in range(num_runs):
            try:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                run_start = time.time()
                output = generator(
                    prompt_text,
                    max_new_tokens=256,
                    truncation=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                latencies.append(time.time() - run_start)
                successes += 1
            except Exception as e:
                print(f"Inference error: {str(e)}")
                continue
        metrics["avg_latency"] = np.mean(latencies) * 1000 if latencies else 0
        metrics["throughput"] = successes / max(0.001, time.time() - start_time)
        metrics["success_rate"] = successes / num_runs
        if torch.cuda.is_available():
            metrics["memory_usage"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            metrics["memory_usage"] = 0
    except Exception as e:
        print(f"Performance measurement failed: {str(e)}")
    return metrics

def evaluate_execution_success(exec_result, task_type, lib_usage):
    if exec_result["status"] != "success":
        return False
    env = exec_result.get("env", {})
    if task_type == "classification" and not lib_usage.get("sklearn", False):
        return False
    if task_type == "clustering" and not lib_usage.get("sklearn", False):
        return False
    if task_type == "scientific" and not any(lib_usage.values()):
        return False
    if task_type == "classification":
        return "accuracy" in env or "Accuracy" in str(env)
    elif task_type == "clustering":
        return "KMeans" in str(env) and "clusters" in str(env)
    elif task_type == "scientific":
        return "results" in str(env) or "analysis" in str(env)
    elif task_type == "problem_solving":
        return "solution" in str(env) or "answer" in str(env)
    return "print" in str(env) and "=" in str(env)

def run_test_pipeline():
    test_prompts = load_test_datasets()
    num_test_samples = len(test_prompts)
    results = {}
    for technique in OPTIMIZATION_TECHNIQUES:
        print(f"\n{'='*40}")
        print(f"Testing: {technique}")
        print(f"{'='*40}")
        model = apply_optimization(technique)
        if model is None:
            print(f"Skipping {technique} due to error")
            continue
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() and not CPU_MODE else "cpu")
        model.to(device)
        print(f"Using device: {device}")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
        tech_results = {
            "inference": {
                "avg_latency": 0,
                "throughput": 0,
                "memory_usage": 0,
                "success_rate": 0
            },
            "quality": {
                "syntax_errors": 0,
                "execution_errors": 0,
                "execution_success": 0,
                "valid_count": 0,
                "quality_score": 0
            }
        }
        perf_metrics = measure_inference_performance(generator, test_prompts[0]["text"])
        tech_results["inference"] = perf_metrics
        for item in tqdm(test_prompts, desc="Testing"):
            try:
                generated = generate_robust_code(generator, item["text"], item.get("type", ""))
                code, lib_usage = validate_code(generated)
                if not code:
                    tech_results["quality"]["syntax_errors"] += 1
                    continue
                data = item.get("data", None)
                exec_result = safe_execute(code, data)
                if exec_result["status"] == "error":
                    tech_results["quality"]["execution_errors"] += 1
                else:
                    tech_results["quality"]["valid_count"] += 1
                    if evaluate_execution_success(exec_result, item.get("type", ""), lib_usage):
                        tech_results["quality"]["execution_success"] += 1
            except Exception as e:
                tech_results["quality"]["syntax_errors"] += 1
                print(f"Test error: {str(e)}")
        tech_results["quality"]["quality_score"] = (tech_results["quality"]["execution_success"] / max(1, num_test_samples))
        results[technique] = tech_results
    return results, num_test_samples

def present_results(results, num_test_samples):
    table_data = []
    for tech, metrics in results.items():
        inf = metrics["inference"]
        qual = metrics["quality"]
        table_data.append({
            "Technique": tech,
            "Latency (ms)": inf['avg_latency'],
            "Throughput (samples/s)": inf['throughput'],
            "Memory (MB)": inf['memory_usage'],
            "Inference Success (%)": inf['success_rate'] * 100,
            "Syntax Errors": qual["syntax_errors"],
            "Execution Errors": qual["execution_errors"],
            "Execution Success (%)": qual['execution_success'] / num_test_samples * 100,
            "Quality Score": qual['quality_score'] * 100
        })
    results_df = pd.DataFrame(table_data)
    formatted_df = results_df.copy()
    formatted_df["Latency (ms)"] = formatted_df["Latency (ms)"].apply(lambda x: f"{x:.1f}")
    formatted_df["Throughput (samples/s)"] = formatted_df["Throughput (samples/s)"].apply(lambda x: f"{x:.2f}")
    formatted_df["Memory (MB)"] = formatted_df["Memory (MB)"].apply(lambda x: f"{x:.1f}")
    formatted_df["Inference Success (%)"] = formatted_df["Inference Success (%)"].apply(lambda x: f"{x:.1f}")
    formatted_df["Execution Success (%)"] = formatted_df["Execution Success (%)"].apply(lambda x: f"{x:.1f}")
    formatted_df["Quality Score"] = formatted_df["Quality Score"].apply(lambda x: f"{x:.1f}")
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)
    print(formatted_df.to_string(index=False))
    results_filename = f"optimization_results_{timestamp}.csv"
    results_path = os.path.join(results_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    if not results_df.empty:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        results_df.plot.bar(x="Technique", y="Latency (ms)", ax=ax[0, 0], legend=False, color='skyblue')
        ax[0, 0].set_title('Inference Latency')
        ax[0, 0].set_ylabel('Milliseconds')
        results_df.plot.bar(x="Technique", y="Memory (MB)", ax=ax[0, 1], legend=False, color='lightgreen')
        ax[0, 1].set_title('Memory Usage')
        ax[0, 1].set_ylabel('MB')
        results_df.plot.bar(x="Technique", y="Execution Success (%)", ax=ax[1, 0], legend=False, color='salmon')
        ax[1, 0].set_title('Execution Success Rate')
        ax[1, 0].set_ylabel('Percentage')
        results_df.plot.bar(x="Technique", y="Quality Score", ax=ax[1, 1], legend=False, color='purple')
        ax[1, 1].set_title('Code Quality Score')
        ax[1, 1].set_ylabel('Score')
        plt.tight_layout()
        viz_filename = f"optimization_results_{timestamp}.png"
        viz_path = os.path.join(results_dir, viz_filename)
        plt.savefig(viz_path, dpi=150)
        print(f"Visualization saved to {viz_path}")
        plt.close()
    return results_df

if __name__ == "__main__":
    print(f"Starting optimization comparison with {TEST_SAMPLE_SIZE} test prompts")
    torch.manual_seed(42)
    np.random.seed(42)
    results, num_test_samples = run_test_pipeline()
    results_df = present_results(results, num_test_samples)
    metrics_filename = f"detailed_metrics_{timestamp}.json"
    metrics_path = os.path.join(results_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed metrics saved to {metrics_path}")
    print("\nTest pipeline completed successfully!")

# %%
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import ast
import json
import os
import subprocess
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_blobs
from datasets import load_dataset, Dataset
import torch.nn.utils.prune as prune
import sklearn
import datetime
import gc
from peft import PeftModel
from tqdm import tqdm

CPU_MODE = False
TEST_SAMPLE_SIZE = 5

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"optimization_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"All results will be saved in: {results_dir}")

base_model_name = "bigcode/tiny_starcoder_py"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=None
)
model = PeftModel.from_pretrained(base_model, "./codegen-lora-adapters")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./final-model")
tokenizer.save_pretrained("./final-model")

def load_test_datasets():
    test_prompts = []
    try:
        scidocs_path = "scidocs_data"
        os.makedirs(scidocs_path, exist_ok=True)
        if not os.path.exists(os.path.join(scidocs_path, "paper_metadata_view_cite_read.json")):
            subprocess.run([
                "aws", "s3", "sync", "--no-sign-request",
                "s3://ai2-s2-research-public/specter/scidocs/",
                scidocs_path, "--region", "us-west-2", "--quiet"
            ], check=True)
        with open(os.path.join(scidocs_path, "paper_metadata_view_cite_read.json"), "r") as f:
            scidocs_data = json.load(f)
        for i, (paper_id, content) in enumerate(scidocs_data.items()):
            if i >= 2: break
            title = content.get('title', '') or ''
            abstract = content.get('abstract', '') or ''
            if len(title) > 10 and len(abstract) > 200:
                test_prompts.append({
                    "text": (
                        f"Generate complete Python code for: {title}\n"
                        f"Abstract: {abstract[:300]}\n"
                        "Create a synthetic dataset and implement analysis."
                    ),
                    "source": "scidocs",
                    "type": "scientific"
                })
    except Exception as e:
        print(f"SciDocs loading failed: {str(e)}")
    for i in range(3):
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=i)
        data = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(4)])
        data["target"] = y
        test_prompts.append({
            "text": (
                "Create a RandomForest classifier with train-test split and show accuracy\n"
                "The input data is in a DataFrame 'df' with features and 'target' column\n"
                "Steps:\n"
                "1. Split into features (X) and target (y)\n"
                "2. Create train/test splits\n"
                "3. Train classifier\n"
                "4. Make predictions\n"
                "5. Print accuracy"
            ),
            "data": data,
            "source": "synthetic",
            "type": "classification"
        })
    print(f"Created {len(test_prompts)} test prompts")
    return test_prompts

def generate_robust_code(generator, prompt_text, task_type):
    task_instructions = {
        "scientific": (
            "Implement complete scientific analysis using numpy/pandas\n"
            "Requirements:\n"
            "1. Create synthetic dataset\n"
            "2. Perform meaningful calculations\n"
            "3. Print clear results\n"
            "4. DO NOT just import libraries without using them"
        ),
        "classification": (
            "Use RandomForestClassifier with train-test split\n"
            "Requirements:\n"
            "1. Split data into features (X) and target (y)\n"
            "2. Create train/test splits\n"
            "3. Train classifier\n"
            "4. Make predictions\n"
            "5. Print accuracy\n"
            "6. DO NOT just import libraries without using them"
        )
    }.get(task_type, "Implement complete solution with meaningful operations")
    
    structured_prompt = f"""
Generate complete, self-contained Python code to solve:
{prompt_text}

Specific Instructions:
{task_instructions}

Code must:
- Use ONLY numpy, pandas, sklearn, matplotlib
- Print results clearly
- For visualizations: plt.savefig('output.png')
- Have NO unused imports
- Be syntactically correct

Code:
```python
"""
    try:
        output = generator(
            structured_prompt,
            temperature=0.1,
            max_new_tokens=512,
            truncation=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95
        )
        return output[0]['generated_text']
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return ""

def validate_code(generated_code):
    if not generated_code:
        return "", {"numpy": False, "pandas": False, "sklearn": False, "matplotlib": False}
    if "```python" in generated_code:
        code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        code = generated_code.split("```")[1].split("```")[0]
    else:
        code = generated_code
    lib_usage = {
        "numpy": False,
        "pandas": False,
        "sklearn": False,
        "matplotlib": False
    }
    repairs = [
        (r"from\s+sklearn\s+import\s+\*", 
         "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.cluster import KMeans"),
        (r"classifier\.fit\(\)", "classifier.fit(X_train, y_train)"),
        (r"predict\(\)", "predict(X_test)"),
        (r"plt\.show\(\)", "plt.savefig('output.png')"),
        (r"import matplotlib\.pyplot as plt", 
         "import matplotlib.pyplot as plt\nplt.switch_backend('Agg')"),
        (r"\.to_csv\('data\.csv'\)", "")
    ]
    for pattern, replacement in repairs:
        code = re.sub(pattern, replacement, code)
    if "np." in code or "numpy." in code:
        lib_usage["numpy"] = True
    if "pd." in code or "pandas." in code:
        lib_usage["pandas"] = True
    if "train_test_split" in code or "RandomForestClassifier" in code or "KMeans" in code:
        lib_usage["sklearn"] = True
    if "plt." in code or "matplotlib." in code:
        lib_usage["matplotlib"] = True
    required_imports = []
    if lib_usage["numpy"]:
        required_imports.append("import numpy as np")
    if lib_usage["pandas"]:
        required_imports.append("import pandas as pd")
    if lib_usage["matplotlib"]:
        required_imports.append("import matplotlib.pyplot as plt")
    if lib_usage["sklearn"]:
        required_imports.append("from sklearn.ensemble import RandomForestClassifier")
        required_imports.append("from sklearn.model_selection import train_test_split")
        required_imports.append("from sklearn.metrics import accuracy_score")
    for imp in required_imports:
        if imp not in code:
            code = imp + "\n" + code
    code_lines = code.split('\n')
    cleaned_lines = []
    for line in code_lines:
        if line.strip().startswith("import") or line.strip().startswith("from"):
            lib_name = line.split()[1].split(".")[0] if "import" in line else line.split()[1]
            if lib_name in ["numpy", "np", "pandas", "pd", "sklearn", "matplotlib", "plt"]:
                if any(f"{lib_name}." in line for line in code_lines):
                    cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines), lib_usage

def safe_execute(code: str, data=None):
    if not code:
        return {"status": "error", "message": "Empty code"}
    safe_env = {
        "__builtins__": {
            'print': print, 'range': range, 'len': len, 'str': str, 'int': int, 
            'float': float, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 
            'set': set, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'enumerate': enumerate, 'zip': zip
        },
        "np": np,
        "pd": pd,
        "plt": plt,
        "RandomForestClassifier": RandomForestClassifier,
        "KMeans": KMeans,
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
    }
    if data is not None:
        safe_env["df"] = data
    try:
        ast.parse(code)
        exec(code, safe_env)
        return {"status": "success", "env": safe_env}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {str(e)}"}

def run_demo():
    test_prompts = load_test_datasets()
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() and not CPU_MODE else "cpu")
    model = AutoModelForCausalLM.from_pretrained("./final-model")
    model.to(device)
    model.eval()
    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device=device
    )
    print("\n" + "="*60)
    print(f"Testing Base Model on {len(test_prompts)} Prompts")
    print("="*60)
    for i, item in enumerate(test_prompts):
        print(f"\n\n{'='*40}")
        print(f"PROMPT {i+1}/{len(test_prompts)} [{item['source']}]")
        print(f"Type: {item.get('type', 'N/A')}")
        print(f"Content:\n{item['text'][:500]}...")
        print('-'*40)
        generated = generate_robust_code(generator, item["text"], item.get("type", ""))
        code, lib_usage = validate_code(generated)
        print("\nGENERATED CODE:")
        print('-'*40)
        print(code)
        print('-'*40)
        data = item.get("data", None)
        exec_result = safe_execute(code, data)
        result = {
            "prompt_id": i+1,
            "source": item["source"],
            "type": item.get("type", ""),
            "generated_code": code,
            "execution_status": exec_result["status"],
            "lib_usage": lib_usage
        }
        if exec_result["status"] == "success":
            print("\nEXECUTION SUCCESS!")
            print("Output captured in environment")
            result["output"] = "Execution completed successfully"
        else:
            print(f"\nEXECUTION ERROR: {exec_result['message']}")
            result["error"] = exec_result["message"]
        results.append(result)
        print(f"\nCompleted prompt {i+1}/{len(test_prompts)}")
    results_filename = f"base_model_results_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*60)
    print(f"Test completed! Results saved to {results_path}")
    print("="*60)
    print("\nEXECUTION SUMMARY:")
    print('-'*60)
    success_count = sum(1 for r in results if r["execution_status"] == "success")
    print(f"Successful executions: {success_count}/{len(results)}")
    for i, r in enumerate(results):
        status = "SUCCESS" if r["execution_status"] == "success" else f"ERROR: {r.get('error', 'Unknown')}"
        print(f"Prompt {i+1}: {r['source']} ({r['type']}) -> {status}")
    return results

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    results = run_demo()



