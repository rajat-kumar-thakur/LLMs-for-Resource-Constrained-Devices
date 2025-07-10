# 🚀 SLMs for Specific Objectives on Resource-Constrained Devices

## 📝 Overview

This project focuses on developing **small language models (SLMs)** tailored for specific tasks, such as interpreting machine logs, optimized to run on devices with limited resources. The goal is to enable both training and inference on these constrained devices, making advanced language processing accessible in edge computing scenarios.

## 📁 Repository Structure

The repository ([GitHub](https://github.com/rajat-kumar-thakur/LLMs-for-Resource-Constrained-Devices)) currently includes the following files:

- **.gitattributes**: Configures Git attributes for consistent file handling across systems.
- **README.md**: This file, providing a comprehensive project overview.
- **desc.txt**: A text file summarizing the project description, datasets, models, optimization techniques, and results.

Additional directories likely include:

- **data/**: Stores datasets like Wikipedia Field of Science, Astronomy Stack DPO Text, SciDocs, and The Stack.
- **models/**: Contains pre-trained models such as Tiny StarCoder Py and TinyLlama 1.1B.
- **notebooks/**: Holds Jupyter notebooks for data preprocessing, model training, and result analysis.
- **results/**: Stores experiment outputs, including logs, plots, and performance metrics.
- **scripts/**: Includes Python scripts for data loading, model training, evaluation, and deployment.

## 📚 Datasets

The following datasets were used for training and testing:

- [Wikipedia Field of Science](https://huggingface.co/datasets/millawell/wikipedia_field_of_science): 📖 Categorizes Wikipedia articles by scientific field, ideal for scientific text processing.
- [Astronomy Stack DPO Text](https://huggingface.co/datasets/David-Xu/astronomy-stack-dpo-text): 🌌 Astronomy-related discussion data for domain-specific tasks.
- [SciDocs](https://github.com/allenai/scidocs): 📄 Supports scientific document understanding tasks like classification.
- [The Stack](https://huggingface.co/datasets/bigcode/the-stack): 💻 A large collection of source code for code-related tasks.

## 🤖 Models

Two models were employed:

- [Tiny StarCoder Py](https://huggingface.co/bigcode/tiny_starcoder_py): 🐍 A compact model for Python code generation and understanding.
- [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T): 🦙 A 1.1 billion parameter model suited for resource-constrained environments.

## ⚙️ Optimization Techniques

To optimize for resource-constrained devices, the following techniques were applied:

- **Pruning**: ✂️ Reduces model size by removing less important weights, used for both TinyLlama and StarCoder.
- **Weight Sharing**: 🔄 Shares weights to reduce memory usage, applied only to StarCoder.
- **Early Exit**: 🚪 Allows early predictions to save computation time, applied only to StarCoder.

## 📊 Results

The performance of the models was evaluated across multiple metrics.

### TinyLlama

| Technique  | Latency (ms) | Throughput (samples/s) | Memory (MB) | Inference Success (%) | Valid Code (%) | Execution Success (%) | Quality Score |
|------------|--------------|------------------------|-------------|------------------------|----------------|------------------------|---------------|
| Base Model | 10057.68     | 0.10                   | 4208.8      | 100.0                  | 43.6           | 56.9                   | 65.6          |
| Pruning    | 5402.01      | 0.19                   | 4209.3      | 100.0                  | 39.6           | 47.5                   | 62.0          |

### StarCoder

| Technique      | Latency (ms) | Throughput (samples/s) | Memory (MB) | Inference Success (%) | Syntax Errors |
|----------------|--------------|------------------------|-------------|------------------------|---------------|
| Base Model     | 2664.09      | 0.3425                 | 2080.48     | 100                    | 5             |
| Pruning        | 2737.22      | 0.3334                 | 2080.61     | 100                    | 0             |
| Weight Sharing | 2711.61      | 0.3387                 | 2080.61     | 100                    | 1             |
| Early Exit     | 2571.31      | 0.3560                 | 2081.48     | 100                    | 3             |

## 🔍 Analysis

The results highlight the impact of optimization techniques:

- **TinyLlama**: Pruning significantly reduced latency (from 10057.68 ms to 5402.01 ms) and increased throughput (from 0.10 to 0.19 samples/s). Memory usage remained nearly unchanged (4208.8 MB to 4209.3 MB), with slight decreases in valid code (43.6% to 39.6%), execution success (56.9% to 47.5%), and quality score (65.6 to 62.0).

- **StarCoder**: The updated results show consistent memory usage across all techniques (~2080 MB), with a negligible increase for early exit (2081.48 MB), suggesting limited memory reduction from current optimizations.
  - **Latency**: Early exit reduces latency (from 2664.09 ms to 2571.31 ms), aligning with its goal of faster predictions. Pruning and weight sharing slightly increase latency (to 2737.22 ms and 2711.61 ms), possibly due to implementation overheads.
  - **Throughput**: Early exit improves throughput (from 0.3425 to 0.3560 samples/s), while pruning and weight sharing slightly reduce it (to 0.3334 and 0.3387 samples/s).
  - **Syntax Errors**: Pruning eliminates all syntax errors (from 5 to 0), significantly improving code quality. Weight sharing and early exit reduce errors to 1 and 3, respectively, still better than the base model.

These findings suggest early exit is most effective for StarCoder’s inference speed, while pruning excels in improving code quality. The consistent memory usage indicates a need for further optimization to reduce the model’s footprint.

## 🎯 Conclusion

This project demonstrates the feasibility of deploying SLMs on resource-constrained devices for tasks like machine log interpretation. TinyLlama benefits from pruning’s latency reduction, despite minor code quality trade-offs. StarCoder sees improved inference speed with early exit and enhanced code quality with pruning. However, the consistent memory usage across StarCoder optimizations highlights the need for further research to minimize resource demands.

## 🛠️ Installation

To set up the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rajat-kumar-thakur/LLMs-for-Resource-Constrained-Devices.git
   cd LLMs-for-Resource-Constrained-Devices
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `requirements.txt` is absent, install packages like `transformers` and `torch` manually.*

## 💻 Usage

To perform inference with the models:

1. **Load the model**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"  # or "bigcode/tiny_starcoder_py"
   model = AutoModelForCausalLM.from_pretrained(model_name)
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```

2. **Perform inference**:
   ```python
   input_text = "Your input text here"
   inputs = tokenizer(input_text, return_tensors="pt")
   outputs = model.generate(**inputs)
   print(tokenizer.decode(outputs[0]))
   ```

*For specific tasks like machine log interpretation, check scripts or notebooks in the repository (if available).*

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bugfix.
3. **Submit a pull request** with a clear description of your changes.
4. **Report issues** or suggest features via the [issue tracker](https://github.com/rajat-kumar-thakur/LLMs-for-Resource-Constrained-Devices/issues).

Ensure your code follows the project’s coding standards and includes tests.

## 🔮 Future Work

Future efforts will focus on:

- Developing strategies to reduce StarCoder’s memory usage, as current optimizations show minimal impact.
- Expanding the dataset with diverse machine log data to improve model robustness.
- Exploring additional techniques like quantization or knowledge distillation to balance latency, throughput, and memory usage.

These efforts aim to enhance SLM efficiency for broader deployment on resource-constrained devices.

## 📄 License

This project is licensed under the [MIT License](LICENSE).