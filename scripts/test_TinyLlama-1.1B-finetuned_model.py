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
from torch.quantization import quantize_dynamic
import sklearn
import types
import tempfile

CPU_MODE = False
TEST_SAMPLE_SIZE = 15
OPTIMIZATION_TECHNIQUES = [
    "Base Model",
    "Pruning",
    "Quantization",
    "Weight Sharing"
]

tokenizer = AutoTokenizer.from_pretrained("./final-model")

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
            if i >= 5:
                break
            title = content.get('title', '') or ''
            abstract = content.get('abstract', '') or ''
            if len(title) > 10 and len(abstract) > 200:
                test_prompts.append({
                    "text": (
                        f"Generate a complete, self-contained Python code for text classification. "
                        f"Title: {title}\n"
                        f"Abstract: {abstract[:400]}\n"
                        "Create a synthetic dataset based on the abstract and implement a classification model."
                    ),
                    "source": "scidocs",
                    "type": "classification"
                })
    except Exception as e:
        print(f"SciDocs loading failed: {str(e)}")
    try:
        astronomy = load_dataset("David-Xu/astronomy-stack-dpo-text", split="train")
        for i, example in enumerate(astronomy):
            if i >= 5:
                break
            test_prompts.append({
                "text": (
                    "Generate a complete, self-contained Python code to solve this astronomy problem:\n"
                    f"{example['prompt']}\n"
                    "Create any necessary synthetic data and implement a solution."
                ),
                "source": "astronomy",
                "type": "problem_solving"
            })
    except Exception as e:
        print(f"Astronomy dataset loading failed: {str(e)}")
    try:
        science = load_dataset("millawell/wikipedia_field_of_science", split="train")
        for i, example in enumerate(science):
            if i >= 5:
                break
            test_prompts.append({
                "text": (
                    "Generate a complete, self-contained Python code for scientific text classification:\n"
                    f"Text: {example['text']}\n"
                    "Create a synthetic dataset and implement a classification model."
                ),
                "source": "wikipedia_science",
                "type": "classification"
            })
    except Exception as e:
        print(f"Science dataset loading failed: {str(e)}")
    for i in range(5):
        X, y = make_classification(
            n_samples=100, 
            n_features=4, 
            n_informative=2, 
            n_classes=2,
            random_state=i
        )
        data = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(4)])
        data["target"] = y
        test_prompts.append({
            "text": "Create a RandomForest classifier and show accuracy",
            "data": data,
            "source": "synthetic",
            "type": "classification"
        })
        X, y = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=i)
        data = pd.DataFrame(X, columns=["x", "y"])
        test_prompts.append({
            "text": "Perform K-means clustering on this data",
            "data": data,
            "source": "synthetic",
            "type": "clustering"
        })
        X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=i)
        outliers = np.random.uniform(low=-10, high=10, size=(5, 2))
        X = np.vstack([X, outliers])
        data = pd.DataFrame(X, columns=["x", "y"])
        test_prompts.append({
            "text": "Detect anomalies using Isolation Forest",
            "data": data,
            "source": "synthetic",
            "type": "outlier_detection"
        })
    print(f"Created {len(test_prompts)} test prompts")
    return test_prompts

def apply_optimization(technique_name):
    try:
        model = AutoModelForCausalLM.from_pretrained("./final-model")
        if technique_name == "Pruning":
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and "lora" not in name.lower():
                    try:
                        prune.l1_unstructured(module, name='weight', amount=0.1)
                        prune.remove(module, 'weight')
                    except:
                        continue
            return model
        if technique_name == "Quantization":
            model = model.cpu()
            return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        if technique_name == "Weight Sharing":
            if hasattr(model, 'lm_head') and hasattr(model, 'model'):
                if hasattr(model.model, 'embed_tokens'):
                    try:
                        model.lm_head.weight = model.model.embed_tokens.weight
                    except:
                        pass
            return model
        return model
    except Exception as e:
        print(f"Error applying {technique_name}: {str(e)}")
        return None

def generate_robust_code(generator, prompt_text, task_type):
    if task_type == "classification":
        task_instructions = "Focus on classification using RandomForestClassifier. Create synthetic data if needed."
    elif task_type == "clustering":
        task_instructions = "Use KMeans clustering and visualize results with matplotlib."
    elif task_type == "outlier_detection":
        task_instructions = "Use IsolationForest for outlier detection. Highlight anomalies in visualization."
    elif task_type == "problem_solving":
        task_instructions = "Solve the problem using appropriate scientific computing techniques."
    else:
        task_instructions = "Solve the problem efficiently with appropriate algorithms."
    structured_prompt = f"""
Generate complete, self-contained Python code to solve this task:
{prompt_text}

Specific Requirements:
1. Create any necessary synthetic data if not provided
2. Use only numpy, pandas, sklearn and matplotlib
3. {task_instructions}
4. Create complete, runnable code
5. Print results clearly
6. For visualizations, use plt.savefig('output.png') instead of plt.show()
7. Ensure the code is syntactically correct

Code:
```python
"""
    try:
        output = generator(
            structured_prompt,
            temperature=0.1,
            max_new_tokens=700,
            truncation=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        return output[0]['generated_text']
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return ""

def validate_code(generated_code):
    if not generated_code:
        return ""
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]
    repairs = [
        (r"from sklearn\.\w+ import \*", ""),
        (r"fit\(\)", "fit(X_train, y_train)"),
        (r"predict\(\)", "predict(X_test)"),
        (r"plt\.show\(\)", "plt.savefig('output.png')"),
        (r"import matplotlib\.pyplot as plt", "import matplotlib.pyplot as plt\nplt.switch_backend('Agg')"),
        (r"\.to_csv\('data\.csv'\)", "")
    ]
    for pattern, replacement in repairs:
        generated_code = re.sub(pattern, replacement, generated_code)
    required_imports = [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt"
    ]
    for imp in required_imports:
        if imp not in generated_code:
            generated_code = imp + "\n" + generated_code
    if "from sklearn" not in generated_code and "import sklearn" not in generated_code:
        generated_code = "from sklearn.ensemble import RandomForestClassifier, IsolationForest\n" + \
                         "from sklearn.cluster import KMeans\n" + \
                         "from sklearn.model_selection import train_test_split\n" + \
                         "from sklearn.metrics import accuracy_score, classification_report\n" + generated_code
    if "pd.DataFrame" not in generated_code and "X =" not in generated_code:
        synthetic_data = "\n# Create synthetic data\nX = np.random.rand(100, 4)\ny = np.random.randint(0, 2, 100)\n"
        generated_code = generated_code.replace("import numpy as np", "import numpy as np" + synthetic_data, 1)
    return generated_code.strip()

def safe_execute(code: str, data=None):
    if not code:
        return {"status": "error", "message": "Empty code"}
    safe_env = {
        "__builtins__": {
            'print': print, 'range': range, 'len': len, 'str': str, 'int': int,
            'float': float, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
            'set': set, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'enumerate': enumerate, 'zip': zip, '__import__': __import__
        },
        "np": np,
        "pd": pd,
        "plt": plt,
        "sklearn": sklearn,
        "RandomForestClassifier": RandomForestClassifier,
        "IsolationForest": IsolationForest,
        "KMeans": KMeans,
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
    }
    if data is not None:
        safe_env["data"] = data
    try:
        ast.parse(code)
        exec(code, safe_env)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {str(e)}"}

def measure_inference_performance(generator, prompt_text, num_runs=3):
    metrics = {
        "avg_latency": 0,
        "throughput": 0,
        "memory_usage": 0,
        "success_rate": 0
    }
    successes = 0
    latencies = []
    try:
        _ = generator(prompt_text, max_new_tokens=50, truncation=True)
        start_time = time.time()
        for _ in range(num_runs):
            try:
                run_start = time.time()
                output = generator(
                    prompt_text,
                    max_new_tokens=300,
                    truncation=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                latencies.append(time.time() - run_start)
                successes += 1
            except Exception:
                continue
        metrics["avg_latency"] = np.mean(latencies) * 1000 if latencies else 0
        metrics["throughput"] = successes / max(0.001, time.time() - start_time)
        metrics["success_rate"] = successes / num_runs
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            metrics["memory_usage"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            metrics["memory_usage"] = 0
    except Exception as e:
        print(f"Performance measurement failed: {str(e)}")
    return metrics

def evaluate_code_quality(generated_code):
    if not generated_code:
        return {
            "has_imports": False,
            "has_model": False,
            "has_print": False,
            "is_runnable": False,
            "score": 0
        }
    has_imports = any(keyword in generated_code for keyword in
                      ["import numpy", "import pandas", "import sklearn"])
    has_model = any(keyword in generated_code for keyword in
                    ["RandomForest", "IsolationForest", "KMeans"])
    has_print = "print(" in generated_code
    has_visualization = "plt.savefig" in generated_code or "plt.plot" in generated_code
    has_data = "X =" in generated_code or "pd.DataFrame" in generated_code
    is_runnable = has_imports and has_model and has_print and has_data
    score = sum([has_imports, has_model, has_print, is_runnable, has_visualization]) / 5
    return {
        "has_imports": has_imports,
        "has_model": has_model,
        "has_print": has_print,
        "has_visualization": has_visualization,
        "has_data": has_data,
        "is_runnable": is_runnable,
        "score": score
    }

def run_smoke_tests(generator, test_prompts):
    print("\n" + "="*50)
    print("Running Enhanced Smoke Tests")
    print("="*50)
    smoke_prompts = []
    for prompt in test_prompts:
        if prompt["source"] in ["scidocs", "astronomy", "wikipedia_science"]:
            smoke_prompts.append(prompt)
            if len(smoke_prompts) >= 3:
                break
    for prompt in test_prompts:
        if prompt["source"] == "synthetic":
            smoke_prompts.append(prompt)
            break
    for i, item in enumerate(smoke_prompts):
        print(f"\nSmoke Test {i+1}: {item['text'][:100]}...")
        generated = generate_robust_code(generator, item["text"], item.get("type", ""))
        code = validate_code(generated)
        print("\nGenerated Code:")
        print(code[:1000] + "..." if len(code) > 1000 else code)
        data = item.get("data", None)
        exec_result = safe_execute(code, data)
        print("\nExecution Result:")
        print(exec_result)
        if exec_result["status"] == "error":
            print("\nFULL CODE WITH ERROR:")
            print(code)
        if os.path.exists("output.png"):
            print("Visualization created: output.png")
            os.remove("output.png")
        print("-"*50)

def run_test_pipeline():
    test_prompts = load_test_datasets()
    results = {}
    for technique in OPTIMIZATION_TECHNIQUES:
        print(f"\n{'='*40}")
        print(f"Testing: {technique}")
        print(f"{'='*40}")
        model = apply_optimization(technique)
        if model is None:
            print(f"Skipping {technique} due to initialization error")
            continue
        model.eval()
        device = 0 if torch.cuda.is_available() and not CPU_MODE else -1
        if technique == "Quantization":
            device = -1
        print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        if technique == "Base Model":
            run_smoke_tests(generator, test_prompts)
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
                "valid_count": 0,
                "quality_score": 0,
                "scores": []
            }
        }
        if test_prompts:
            perf_metrics = measure_inference_performance(generator, test_prompts[0]["text"])
            tech_results["inference"] = perf_metrics
        for item in tqdm(test_prompts, desc="Testing"):
            try:
                generated = generate_robust_code(generator, item["text"], item.get("type", ""))
                code = validate_code(generated)
                if not code:
                    tech_results["quality"]["syntax_errors"] += 1
                    tech_results["quality"]["scores"].append(0)
                    continue
                quality_metrics = evaluate_code_quality(code)
                quality_score = quality_metrics["score"]
                tech_results["quality"]["scores"].append(quality_score)
                if not quality_metrics["is_runnable"]:
                    tech_results["quality"]["syntax_errors"] += 1
                    continue
                tech_results["quality"]["valid_count"] += 1
                data = item.get("data", None)
                if data is not None:
                    exec_result = safe_execute(code, data)
                    if exec_result["status"] == "error":
                        tech_results["quality"]["execution_errors"] += 1
            except Exception as e:
                tech_results["quality"]["syntax_errors"] += 1
                tech_results["quality"]["scores"].append(0)
                print(f"Error during testing: {str(e)}")
        if tech_results["quality"]["scores"]:
            tech_results["quality"]["quality_score"] = np.mean(tech_results["quality"]["scores"])
        else:
            tech_results["quality"]["quality_score"] = 0
        results[technique] = tech_results
    return results

def present_results(results):
    table_data = []
    for tech, metrics in results.items():
        inf = metrics["inference"]
        qual = metrics["quality"]
        valid_count = qual["valid_count"]
        total_tests = len(qual["scores"]) if "scores" in qual else TEST_SAMPLE_SIZE
        table_data.append({
            "Technique": tech,
            "Latency (ms)": f"{inf['avg_latency']:.2f}",
            "Throughput (samples/s)": f"{inf['throughput']:.2f}",
            "Memory (MB)": f"{inf['memory_usage']:.1f}",
            "Inference Success (%)": f"{inf['success_rate'] * 100:.1f}",
            "Valid Code (%)": f"{valid_count / total_tests * 100:.1f}" if total_tests > 0 else "N/A",
            "Execution Success (%)": f"{(1 - qual['execution_errors'] / max(1, valid_count)) * 100:.1f}" if valid_count > 0 else "N/A",
            "Quality Score": f"{qual['quality_score'] * 100:.1f}"
        })
    results_df = pd.DataFrame(table_data)
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)
    print(results_df.to_string(index=False))
    results_df.to_csv("optimization_results.csv", index=False)
    print("\nResults saved to optimization_results.csv")
    if not results_df.empty:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        results_df["Latency Value"] = results_df["Latency (ms)"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df["Valid Code Value"] = results_df["Valid Code (%)"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df["Exec Success Value"] = results_df["Execution Success (%)"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df["Quality Value"] = results_df["Quality Score"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df.plot.bar(x="Technique", y="Latency Value", ax=ax[0, 0], legend=False, color='skyblue')
        ax[0, 0].set_title('Inference Latency')
        ax[0, 0].set_ylabel('Milliseconds')
        results_df.plot.bar(x="Technique", y="Valid Code Value", ax=ax[0, 1], legend=False, color='lightgreen')
        ax[0, 1].set_title('Valid Code Rate')
        ax[0, 1].set_ylabel('Percentage')
        results_df.plot.bar(x="Technique", y="Exec Success Value", ax=ax[1, 0], legend=False, color='salmon')
        ax[1, 0].set_title('Execution Success Rate')
        ax[1, 0].set_ylabel('Percentage')
        results_df.plot.bar(x="Technique", y="Quality Value", ax=ax[1, 1], legend=False, color='purple')
        ax[1, 1].set_title('Code Quality Score')
        ax[1, 1].set_ylabel('Score (0-100)')
        plt.tight_layout()
        plt.savefig('optimization_results.png', dpi=150)
        print("Visualization saved to optimization_results.png")
        plt.close()
    return results_df

if __name__ == "__main__":
    print(f"Starting optimization comparison with {TEST_SAMPLE_SIZE} test prompts")
    torch.manual_seed(42)
    np.random.seed(42)
    results = run_test_pipeline()
    results_df = present_results(results)
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

CPU_MODE = False
TEST_SAMPLE_SIZE = 100
OPTIMIZATION_TECHNIQUES = ["Base Model", "Pruning"]

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"optimization_results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)
print(f"All results will be saved in: {results_dir}")

tokenizer = AutoTokenizer.from_pretrained("./final-model")

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
            if i >= 30:
                break
            title = content.get('title', '') or ''
            abstract = content.get('abstract', '') or ''
            if len(title) > 10 and len(abstract) > 200:
                test_prompts.append({
                    "text": (
                        f"Generate a complete, self-contained Python code for text classification. "
                        f"Title: {title}\n"
                        f"Abstract: {abstract[:400]}\n"
                        "Create a synthetic dataset based on the abstract and implement a classification model."
                    ),
                    "source": "scidocs",
                    "type": "classification"
                })
    except Exception as e:
        print(f"SciDocs loading failed: {str(e)}")
    try:
        astronomy = load_dataset("David-Xu/astronomy-stack-dpo-text", split="train")
        for i, example in enumerate(astronomy):
            if i >= 30:
                break
            test_prompts.append({
                "text": (
                    "Generate a complete, self-contained Python code to solve this astronomy problem:\n"
                    f"{example['prompt']}\n"
                    "Create any necessary synthetic data and implement a solution."
                ),
                "source": "astronomy",
                "type": "problem_solving"
            })
    except Exception as e:
        print(f"Astronomy dataset loading failed: {str(e)}")
    try:
        science = load_dataset("millawell/wikipedia_field_of_science", split="train")
        for i, example in enumerate(science):
            if i >= 30:
                break
            test_prompts.append({
                "text": (
                    "Generate a complete, self-contained Python code for scientific text classification:\n"
                    f"Text: {example['text']}\n"
                    "Create a synthetic dataset and implement a classification model."
                ),
                "source": "wikipedia_science",
                "type": "classification"
            })
    except Exception as e:
        print(f"Science dataset loading failed: {str(e)}")
    for i in range(20):
        X, y = make_classification(
            n_samples=100, 
            n_features=4, 
            n_informative=2, 
            n_classes=2,
            random_state=i
        )
        data = pd.DataFrame(X, columns=[f"feature_{j}" for j in range(4)])
        data["target"] = y
        test_prompts.append({
            "text": "Create a RandomForest classifier and show accuracy",
            "data": data,
            "source": "synthetic",
            "type": "classification"
        })
        X, y = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=i)
        data = pd.DataFrame(X, columns=["x", "y"])
        test_prompts.append({
            "text": "Perform K-means clustering on this data",
            "data": data,
            "source": "synthetic",
            "type": "clustering"
        })
        X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.5, random_state=i)
        outliers = np.random.uniform(low=-10, high=10, size=(5, 2))
        X = np.vstack([X, outliers])
        data = pd.DataFrame(X, columns=["x", "y"])
        test_prompts.append({
            "text": "Detect anomalies using Isolation Forest",
            "data": data,
            "source": "synthetic",
            "type": "outlier_detection"
        })
    print(f"Created {len(test_prompts)} test prompts")
    return test_prompts

def apply_optimization(technique_name):
    try:
        model = AutoModelForCausalLM.from_pretrained("./final-model")
        if technique_name == "Pruning":
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and "lora" not in name.lower():
                    try:
                        prune.l1_unstructured(module, name='weight', amount=0.1)
                        prune.remove(module, 'weight')
                    except:
                        continue
            return model
        return model
    except Exception as e:
        print(f"Error applying {technique_name}: {str(e)}")
        return None

def generate_robust_code(generator, prompt_text, task_type):
    if task_type == "classification":
        task_instructions = "Focus on classification using RandomForestClassifier. Create synthetic data if needed."
    elif task_type == "clustering":
        task_instructions = "Use KMeans clustering and visualize results with matplotlib."
    elif task_type == "outlier_detection":
        task_instructions = "Use IsolationForest for outlier detection. Highlight anomalies in visualization."
    elif task_type == "problem_solving":
        task_instructions = "Solve the problem using appropriate scientific computing techniques."
    else:
        task_instructions = "Solve the problem efficiently with appropriate algorithms."
    structured_prompt = f"""
Generate complete, self-contained Python code to solve this task:
{prompt_text}

Specific Requirements:
1. Create any necessary synthetic data if not provided
2. Use only numpy, pandas, sklearn and matplotlib
3. {task_instructions}
4. Create complete, runnable code
5. Print results clearly
6. For visualizations, use plt.savefig('output.png') instead of plt.show()
7. Ensure the code is syntactically correct

Code:
```python
"""
    try:
        output = generator(
            structured_prompt,
            temperature=0.1,
            max_new_tokens=700,
            truncation=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        return output[0]['generated_text']
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return ""

def validate_code(generated_code):
    if not generated_code:
        return ""
    if "```python" in generated_code:
        generated_code = generated_code.split("```python")[1].split("```")[0]
    elif "```" in generated_code:
        generated_code = generated_code.split("```")[1].split("```")[0]
    repairs = [
        (r"from sklearn\.\w+ import \*", ""),
        (r"fit\(\)", "fit(X_train, y_train)"),
        (r"predict\(\)", "predict(X_test)"),
        (r"plt\.show\(\)", "plt.savefig('output.png')"),
        (r"import matplotlib\.pyplot as plt", "import matplotlib.pyplot as plt\nplt.switch_backend('Agg')"),
        (r"\.to_csv\('data\.csv'\)", "")
    ]
    for pattern, replacement in repairs:
        generated_code = re.sub(pattern, replacement, generated_code)
    required_imports = [
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt"
    ]
    for imp in required_imports:
        if imp not in generated_code:
            generated_code = imp + "\n" + generated_code
    if "from sklearn" not in generated_code and "import sklearn" not in generated_code:
        generated_code = "from sklearn.ensemble import RandomForestClassifier, IsolationForest\n" + \
                         "from sklearn.cluster import KMeans\n" + \
                         "from sklearn.model_selection import train_test_split\n" + \
                         "from sklearn.metrics import accuracy_score, classification_report\n" + generated_code
    if "pd.DataFrame" not in generated_code and "X =" not in generated_code:
        synthetic_data = "\nX = np.random.rand(100, 4)\ny = np.random.randint(0, 2, 100)\n"
        generated_code = generated_code.replace("import numpy as np", "import numpy as np" + synthetic_data, 1)
    return generated_code.strip()


def safe_execute(code: str, data=None):
    if not code:
        return {"status": "error", "message": "Empty code"}
    safe_env = {
        "__builtins__": {
            'print': print, 'range': range, 'len': len, 'str': str, 'int': int,
            'float': float, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
            'set': set, 'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'enumerate': enumerate, 'zip': zip, '__import__': __import__
        },
        "np": np,
        "pd": pd,
        "plt": plt,
        "sklearn": sklearn,
        "RandomForestClassifier": RandomForestClassifier,
        "IsolationForest": IsolationForest,
        "KMeans": KMeans,
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
    }
    if data is not None:
        safe_env["data"] = data
    try:
        ast.parse(code)
        exec(code, safe_env)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {str(e)}"}

def measure_inference_performance(generator, prompt_text, num_runs=3):
    metrics = {
        "avg_latency": 0,
        "throughput": 0,
        "memory_usage": 0,
        "success_rate": 0
    }
    successes = 0
    latencies = []
    try:
        _ = generator(prompt_text, max_new_tokens=50, truncation=True)
        start_time = time.time()
        for _ in range(num_runs):
            try:
                run_start = time.time()
                output = generator(
                    prompt_text,
                    max_new_tokens=300,
                    truncation=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                latencies.append(time.time() - run_start)
                successes += 1
            except Exception:
                continue
        metrics["avg_latency"] = np.mean(latencies) * 1000 if latencies else 0
        metrics["throughput"] = successes / max(0.001, time.time() - start_time)
        metrics["success_rate"] = successes / num_runs
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            metrics["memory_usage"] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            metrics["memory_usage"] = 0
    except Exception as e:
        print(f"Performance measurement failed: {str(e)}")
    return metrics

def evaluate_code_quality(generated_code):
    if not generated_code:
        return {
            "has_imports": False,
            "has_model": False,
            "has_print": False,
            "is_runnable": False,
            "score": 0
        }
    has_imports = any(keyword in generated_code for keyword in 
                      ["import numpy", "import pandas", "import sklearn"])
    has_model = any(keyword in generated_code for keyword in 
                    ["RandomForest", "IsolationForest", "KMeans"])
    has_print = "print(" in generated_code
    has_visualization = "plt.savefig" in generated_code or "plt.plot" in generated_code
    has_data = "X =" in generated_code or "pd.DataFrame" in generated_code
    is_runnable = has_imports and has_model and has_print and has_data
    score = sum([has_imports, has_model, has_print, is_runnable, has_visualization]) / 5
    return {
        "has_imports": has_imports,
        "has_model": has_model,
        "has_print": has_print,
        "has_visualization": has_visualization,
        "has_data": has_data,
        "is_runnable": is_runnable,
        "score": score
    }

def run_test_pipeline():
    test_prompts = load_test_datasets()
    results = {}
    for technique in OPTIMIZATION_TECHNIQUES:
        print(f"\n{'='*40}")
        print(f"Testing: {technique}")
        print(f"{'='*40}")
        model = apply_optimization(technique)
        if model is None:
            print(f"Skipping {technique} due to initialization error")
            continue
        model.eval()
        device = 0 if torch.cuda.is_available() and not CPU_MODE else -1
        print(f"Using device: {'GPU' if device >= 0 else 'CPU'}")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
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
                "valid_count": 0,
                "quality_score": 0,
                "scores": []
            }
        }
        if test_prompts:
            perf_metrics = measure_inference_performance(generator, test_prompts[0]["text"])
            tech_results["inference"] = perf_metrics
        for item in tqdm(test_prompts, desc="Testing"):
            try:
                generated = generate_robust_code(generator, item["text"], item.get("type", ""))
                code = validate_code(generated)
                if not code:
                    tech_results["quality"]["syntax_errors"] += 1
                    tech_results["quality"]["scores"].append(0)
                    continue
                quality_metrics = evaluate_code_quality(code)
                quality_score = quality_metrics["score"]
                tech_results["quality"]["scores"].append(quality_score)
                if not quality_metrics["is_runnable"]:
                    tech_results["quality"]["syntax_errors"] += 1
                    continue
                tech_results["quality"]["valid_count"] += 1
                data = item.get("data", None)
                if data is not None:
                    exec_result = safe_execute(code, data)
                    if exec_result["status"] == "error":
                        tech_results["quality"]["execution_errors"] += 1
            except Exception as e:
                tech_results["quality"]["syntax_errors"] += 1
                tech_results["quality"]["scores"].append(0)
                print(f"Error during testing: {str(e)}")
        if tech_results["quality"]["scores"]:
            tech_results["quality"]["quality_score"] = np.mean(tech_results["quality"]["scores"])
        else:
            tech_results["quality"]["quality_score"] = 0
        results[technique] = tech_results
    return results

def present_results(results):
    table_data = []
    for tech, metrics in results.items():
        inf = metrics["inference"]
        qual = metrics["quality"]
        valid_count = qual["valid_count"]
        total_tests = len(qual["scores"]) if "scores" in qual else TEST_SAMPLE_SIZE
        table_data.append({
            "Technique": tech,
            "Latency (ms)": f"{inf['avg_latency']:.2f}",
            "Throughput (samples/s)": f"{inf['throughput']:.2f}",
            "Memory (MB)": f"{inf['memory_usage']:.1f}",
            "Inference Success (%)": f"{inf['success_rate'] * 100:.1f}",
            "Valid Code (%)": f"{valid_count / total_tests * 100:.1f}" if total_tests > 0 else "N/A",
            "Execution Success (%)": f"{(1 - qual['execution_errors'] / max(1, valid_count)) * 100:.1f}" if valid_count > 0 else "N/A",
            "Quality Score": f"{qual['quality_score'] * 100:.1f}"
        })
    results_df = pd.DataFrame(table_data)
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)
    print(results_df.to_string(index=False))
    results_filename = f"optimization_results_{timestamp}.csv"
    results_path = os.path.join(results_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    if not results_df.empty:
        fig, ax = plt.subplots(2, 2, figsize=(15, 12))
        results_df["Latency Value"] = results_df["Latency (ms)"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df["Valid Code Value"] = results_df["Valid Code (%)"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df["Exec Success Value"] = results_df["Execution Success (%)"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df["Quality Value"] = results_df["Quality Score"].str.extract(r'(\d+\.?\d*)').astype(float)
        results_df.plot.bar(x="Technique", y="Latency Value", ax=ax[0, 0], legend=False, color='skyblue')
        ax[0, 0].set_title('Inference Latency')
        ax[0, 0].set_ylabel('Milliseconds')
        results_df.plot.bar(x="Technique", y="Valid Code Value", ax=ax[0, 1], legend=False, color='lightgreen')
        ax[0, 1].set_title('Valid Code Rate')
        ax[0, 1].set_ylabel('Percentage')
        results_df.plot.bar(x="Technique", y="Exec Success Value", ax=ax[1, 0], legend=False, color='salmon')
        ax[1, 0].set_title('Execution Success Rate')
        ax[1, 0].set_ylabel('Percentage')
        results_df.plot.bar(x="Technique", y="Quality Value", ax=ax[1, 1], legend=False, color='purple')
        ax[1, 1].set_title('Code Quality Score')
        ax[1, 1].set_ylabel('Score (0-100)')
        plt.tight_layout()
        viz_filename = f"optimization_results_{timestamp}.png"
        viz_path = os.path.join(results_dir, viz_filename)
        plt.savefig(viz_path, dpi=150)
        print(f"Visualization saved to {viz_path}")
        plt.close()
    return results_df

if __name__ == "__main__":
    print(f"Starting comprehensive optimization comparison with {TEST_SAMPLE_SIZE} test prompts")
    torch.manual_seed(42)
    np.random.seed(42)
    results = run_test_pipeline()
    results_df = present_results(results)
    metrics_filename = f"detailed_metrics_{timestamp}.json"
    metrics_path = os.path.join(results_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Detailed metrics saved to {metrics_path}")
    print("\nTest pipeline completed successfully!")


