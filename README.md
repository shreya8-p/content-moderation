# 🛡️ Real-Time Content Moderation System

> Detect toxic and policy-violating content before it reaches users — powered by fine-tuned DistilBERT + ONNX Runtime + FastAPI.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![DistilBERT](https://img.shields.io/badge/DistilBERT-HuggingFace-yellow)
![Tests](https://img.shields.io/badge/Tests-16%2F18%20Passing-brightgreen)

---

## 📌 Overview

This project replicates a production-grade content moderation pipeline similar to what companies like **Meta, YouTube, and Twitter** use to moderate billions of posts. It classifies text as **toxic or non-toxic** in real time with ~99.86% confidence using a fine-tuned transformer model.

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| Toxicity Detection Confidence | **99.86%** |
| Clean Text Confidence | **99.88%** |
| Test Coverage | **16/18 passing** |
| Batch Support | **Up to 100 texts/request** |
| Categories Detected | **4 (hate speech, harassment, violence, spam)** |

---

## 🏗️ Architecture

```
User Input (text)
      │
      ▼
FastAPI Endpoint (/moderate)
      │
      ▼
Pydantic Validation
      │
      ▼
DistilBERT Tokenizer (max_length=128)
      │
      ▼
ONNX Runtime Session (optimized graph)
      │
      ▼
Softmax → Toxicity Score
      │
      ▼
JSON Response with categories
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/shreya8-p/content-moderation.git
cd content-moderation
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the API server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open Swagger UI
```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

### `POST /moderate` — Single text moderation
```bash
curl -X POST http://localhost:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "I hate you, you worthless idiot!"}'
```

**Response:**
```json
{
  "text": "I hate you, you worthless idiot!",
  "is_toxic": true,
  "toxicity_score": 0.9986,
  "label": "toxic",
  "inference_time_ms": 107.2,
  "categories": {
    "hate_speech": 1.0,
    "harassment": 1.0,
    "violence": 0.399,
    "spam": 0.699
  }
}
```

### `POST /moderate/batch` — Batch moderation (up to 100 texts)
```bash
curl -X POST http://localhost:8000/moderate/batch \
  -H "Content-Type: application/json" \
  -d '["I hate you!", "Good morning!", "You are worthless!", "Have a great day!"]'
```

**Response:**
```json
{
  "results": [
    {"is_toxic": true,  "score": 0.9986, "label": "toxic"},
    {"is_toxic": false, "score": 0.0012, "label": "non_toxic"},
    {"is_toxic": true,  "score": 0.9986, "label": "toxic"},
    {"is_toxic": false, "score": 0.0011, "label": "non_toxic"}
  ],
  "total": 4
}
```

### `GET /health` — Health check
```json
{"status": "healthy", "model": "distilbert-onnx"}
```

---

## 🧠 ML Pipeline

### 1. Data Preprocessing (`src/utils/preprocess.py`)
- Cleans raw social media text (removes URLs, mentions, extra whitespace)
- Builds binary toxic/non-toxic labels
- Splits into 80/10/10 train/val/test sets

### 2. Model Training (`src/model/train.py`)
- Base model: `distilbert-base-uncased` (66M parameters)
- Fine-tuned for binary sequence classification
- Optimizer: AdamW with linear warmup scheduler
- Best model saved based on F1 score

### 3. ONNX Export (`src/model/export_onx.py`)
- Exports fine-tuned PyTorch model to ONNX format
- Applies graph-level optimizations (operator fusion, constant folding)
- ~3x speedup over raw PyTorch inference

---

## 🗂️ Project Structure

```
content-moderation/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI server
│   ├── model/
│   │   ├── train.py             # DistilBERT fine-tuning
│   │   ├── predictor.py         # ONNX inference wrapper
│   │   └── export_onx.py        # PyTorch → ONNX conversion
│   └── utils/
│       ├── preprocess.py        # Data cleaning & splitting
│       └── logger.py            # Structured logging
├── tests/
│   ├── test_api.py              # API integration tests
│   └── test_predictor.py        # Unit tests
├── configs/
│   └── model_config.yaml        # Hyperparameters
├── docker/
│   └── Dockerfile               # Production container
├── scripts/
│   └── run_pipeline.sh          # Full pipeline script
└── requirements.txt
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

```
tests/test_api.py::test_health_endpoint          PASSED
tests/test_api.py::test_moderate_toxic_text      PASSED
tests/test_api.py::test_batch_moderation         PASSED
tests/test_predictor.py::test_toxic_text_detected PASSED
...
16/18 passed
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | DistilBERT (HuggingFace Transformers) |
| Training | PyTorch + AdamW |
| Inference | ONNX Runtime |
| API Server | FastAPI + Uvicorn |
| Validation | Pydantic |
| Data Processing | Pandas + NumPy |
| Evaluation | Scikit-learn |
| Testing | Pytest |
| Container | Docker |

---

## 💡 Why DistilBERT?

- **40% smaller** than BERT with **97% of its performance**
- **60% faster** inference — critical for real-time moderation
- Pre-trained on massive text corpus — understands context, sarcasm, and subtle toxic language
- Fine-tuning takes hours instead of days

---

## 📈 Scaling to Production

In production (e.g. Meta scale):
- Run on **GPU instances** → reduces latency to <50ms
- Use **horizontal scaling** with multiple API workers
- Add **Redis caching** for repeated content
- Implement **A/B testing** between model versions
- Monitor with **Prometheus + Grafana** dashboards

---

## 👩‍💻 Author

**Shreya** — [GitHub](https://github.com/shreya8-p)
