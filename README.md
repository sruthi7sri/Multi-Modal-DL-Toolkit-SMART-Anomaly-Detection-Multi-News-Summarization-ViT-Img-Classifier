# Deep Learning Architectures: Autoencoders, Transformers & LLM Apps

## Overview
This project explores diverse deep learning architectures and their real-world applications, from anomaly detection and text summarization to image classification and spam detection. It includes implementations of **autoencoders**, **transformers from scratch**, and **pretrained LLMs** (like BART, DistilBERT, ViT) and deploys key models using **Streamlit** and **Gradio**.

---

## Methodology

### 01. Autoencoder for Anomaly Detection (Hard Drive Data)
- Used tabular SMART sensor data from Backblaze hard drives.
- Built 3 custom autoencoder models (varying layers, dropout, latent sizes).
- Trained using MSE loss on normal samples (`failure = 0`).
- Set anomaly threshold using reconstruction errors.
- Evaluated using Accuracy, F1, MAE, RMSE, R².

### 02. Transformer from Scratch (Text Classification)
- Implemented positional encoding, multi-head attention, encoder layers using PyTorch.
- Preprocessed text data with tokenization and vocabulary encoding.
- Trained a full Transformer encoder model for classification.
- Tracked performance using accuracy and loss plots.

### 03. Summarization using BART (Multi-News Dataset)
- Fine-tuned `facebook/bart-base` using Hugging Face Transformers.
- Preprocessed long-form documents and summaries using `BartTokenizer`.
- Evaluated with ROUGE, BLEU, BERTScore.
- Deployed via Gradio.

**Live App: [BART Summarizer – Multi-News](https://huggingface.co/spaces/Sruthisri/bart-summarizer-multi-news)**

### 04. Spam Classification with DistilBERT (LLM Probing)
- Extracted embeddings from DistilBERT and trained a classifier head.
- Used Enron Spam dataset.
- Achieved >85% accuracy and F1 using frozen LLM layers.

### 05. Vision Transformer (ViT) for Cats vs Dogs
- Used pretrained ViT model via `timm` and fine-tuned for binary classification.
- Built and deployed an interactive **Streamlit** app.

**Live App: [ViT Cat vs Dog Classifier](https://vit-cat-dog-classifier-sv94-sarojavu.streamlit.app/)**

---

## Real-World Applications
- **Predictive Maintenance**: Autoencoders identify failing components from sensor logs.
- **Enterprise NLP**: Summarization for legal/news content using BART.
- **Cybersecurity**: Spam classification via frozen LLMs.
- **Edge AI**: Vision Transformer deployed via Streamlit for real-time prediction.

---

## Technology Comparison

| Component         | Chosen Technology           | Alternatives        | Rationale                                             |
|------------------|-----------------------------|---------------------|--------------------------------------------------------|
| Framework        | PyTorch, HuggingFace        | TensorFlow          | PyTorch = flexible + great for research               |
| Summarizer Model | BART                        | T5, Pegasus         | Good balance of speed + quality for long texts        |
| Transformer      | Custom PyTorch Transformer  | HuggingFace encoder | Built from scratch for better understanding           |
| Anomaly Model    | FC Autoencoders             | LSTM, VAE           | Best suited for tabular SMART data                    |
| Deployment       | Gradio, Streamlit           | Flask, FastAPI      | Fast, browser-based deployment for ML apps            |

---

## Running the Project
### Setup Environment
```bash
git clone https://github.com/sruthi7sri/deep-learning-autoencoder-transformer-llm-apps.git
cd deep-learning-autoencoder-transformer-llm-apps
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Run Notebooks
```bash
jupyter notebook
```

---
## Project Goals

- Build deep learning models from scratch to understand architectural fundamentals.
- Apply autoencoders for real-world anomaly detection using hardware sensor data.
- Explore transformer internals by implementing attention, positional encoding, and encoder blocks from the ground up.
- Fine-tune a pre-trained BART model for abstractive text summarization on long-form datasets like Multi-News.
- Utilize frozen LLM embeddings (DistilBERT) for binary text classification in spam detection.
- Deploy real-time, browser-based ML apps using Gradio and Streamlit.
- Compare architectural variants (dense, dropout, latent bottlenecks) to evaluate model robustness.
- Demonstrate model performance through visualizations, metrics (F1, ROUGE, BLEU), and interactive tools.

---
## License

© 2025 Sruthisri Venkateswaran. All rights reserved.
Educational use only.

---

## Contact

I'm always open to feedback, collaborations, or new opportunities in deep learning, ML deployment, and NLP research.

- Email: sruthisri7@gmail.com
- For issues, suggestions, or improvements: [Open an Issue](https://github.com/sruthi7sri/deep-learning-autoencoder-transformer-llm-apps/issues)
