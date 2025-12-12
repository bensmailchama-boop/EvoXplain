# EvoXplain: Detecting Mechanistic Non-Identifiability in Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## About

**EvoXplain** is a framework for detecting and quantifying mechanistic non-identifiability in machine learning models—the phenomenon where models achieve identical predictive performance while producing vastly different explanations of *how* they make decisions.

**Author:** Chama Bensmail 
**Status:** Patent-pending (UK provisional patent filed)  
**Publication:** Preprint forthcoming

> [!NOTE]
> This repository contains the research implementation. Commercial licensing inquiries: bensmail.chama@gmail.com

## 🔬 Key Concept

Standard ML validation focuses on **predictive performance** (accuracy, AUC, etc.), but ignores **mechanistic stability**—whether models consistently identify the same decision-making features. EvoXplain reveals that:

- ✅ Models can achieve 99%+ prediction consistency
- ❌ Yet exhibit 90%+ mechanistic entropy (near-maximal instability)
- ⚠️ This affects the following types: Linear, Tree-based, and Deep Learning

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/bensmailchama-boop/EvoXplain/evoxplain.git
cd evoxplain
pip install -r requirements.txt
```

### Run the Demo

```bash
jupyter notebook evoxplain_demo.ipynb
```

Or execute directly:
```python
python -m jupyter execute evoxplain_demo.ipynb
```

### Basic Usage

Simply configure three parameters and run:

```python
# Choose your dataset
DATASET = 'breast_cancer'  # or 'adult_income', 'compas'

# Choose your model
MODEL_TYPE = 'logistic_regression'  # or 'random_forest', 'svm'

# Number of training runs
N_RUNS = 100 # you can choose greater or smaller number
```

That's it! The notebook handles everything else automatically.

## 📊 What You'll See

The analysis produces:

1. **PCA Visualization** - Shows clusters of distinct mechanistic explanations
2. **Hierarchical Dendrogram** - Reveals the structure of mechanistic diversity
3. **Feature Importance Heatmap** - Displays how explanations vary across runs
4. **Mechanistic Entropy Score** - Quantifies explanation stability (0.0 = stable, 1.0 = chaotic)
5. **Feature Consistency Analysis** - Identifies which features are reliably important

### Example Output

```
MECHANISTIC ENTROPY
  Raw entropy (nats): 0.6730
  Normalized entropy: 0.9710
  Stability assessment: LOW - Many alternative mechanisms

Despite 97.4% mean accuracy (σ=0.003), the model exhibits 
2 distinct mechanistic explanations with 97.1% normalized entropy.
```

## 🎯 Use Cases

### For Researchers
- Validate that your model explanations are mechanistically stable
- Compare mechanistic stability across different architectures
- Establish baseline entropy levels for your domain

### For Practitioners
- Ensure regulatory compliance (medical, financial applications)
- Detect when model explanations are unreliable
- Choose models with both high accuracy AND mechanistic stability

### For ML Safety
- Identify models where explanations may mislead human oversight
- Quantify explanation risk in high-stakes applications
- Establish mechanistic validation as a standard practice

## 🔧 Supported Configurations

### Datasets
- **Breast Cancer Wisconsin** - Medical diagnosis (569 samples, 30 features)
- **Adult Income** - Socioeconomic prediction (32k samples, 108 features after encoding)
- **COMPAS** - Recidivism risk assessment (7k samples, ~10 features after encoding)

### Models
- **Logistic Regression** - Tests convex models with regularization
- **Random Forest** - Tests tree ensemble stability
- **Support Vector Machines** - Tests both linear and RBF kernels

### Explanation Methods
- **Gradient × Input** - For linear/logistic models
- **Feature Importance** - For tree-based models
- **Support Vector Attribution** - For SVM models

## 📈 Interpreting Results

### Mechanistic Entropy (Interpretation Guide)

- < 0.30 — High stability: one dominant explanatory mechanism

- 0.30–0.70 — Moderate stability: multiple mechanisms with uneven support

- > 0.70 — Low stability: several competing mechanisms

- > 0.85 — Severe instability: explanations split across incompatible basins

### Consistency Ratio (Feature-Level Stability Guide)

- > 80% — High feature stability: most key features appear consistently across runs

- 50–80% — Moderate stability: some disagreement across runs

- < 50% — Low stability: high variability in which features the model relies on

## 🧪 Extending the Framework

### Add a New Dataset

```python
def load_custom_dataset():
    X = ...  # Load your features
    y = ...  # Load your labels
    # Preprocess as needed
    return X_train, X_test, y_train, y_test, feature_names, dataset_info
```

### Add a New Model

```python
def train_custom_model(X_train, y_train, seed):
    model = YourModel(random_state=seed)
    model.fit(X_train, y_train)
    return model, accuracy, params
```

### Add a New Explanation Method

```python
def extract_custom_explanations(model, X_train):
    importance = your_attribution_method(model, X_train)
    return importance
```

## 📝 Citation

If you use EvoXplain in your research, please cite:



## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Breast Cancer Wisconsin dataset from UCI ML Repository
- Adult Income dataset from UCI ML Repository  
- COMPAS dataset via ProPublica
- Built with scikit-learn, NumPy, and matplotlib
- University of Hertfordshire HPC
  

## 📬 Contact

- **Author**: [Chama Bensmail]
- **Email**: [bensmail.chama@gmail.com]
- **Paper**: []
- **Issues**: [GitHub Issues](https://github.com/bensmailchama-boop/EvoXplain/issues)

---

**⭐ If you find this work useful, please consider starring the repository!**
