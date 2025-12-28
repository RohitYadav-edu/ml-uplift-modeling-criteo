# Uplift Modeling with Serverless Deployment on AWS

This project demonstrates an **end-to-end uplift modeling system**, from causal machine learning in Jupyter notebooks to a **production-style serverless inference API** deployed on AWS using **Lambda container images** and **API Gateway**.

Unlike traditional classification models that predict *who will convert*, uplift models estimate **who will convert *because of a treatment***, enabling more efficient and targeted decision-making.

This repository focuses on both:
- **Machine Learning correctness** (uplift modeling, evaluation)
- **Engineering robustness** (deployment, scalability, validation, APIs)

---

## What is Uplift Modeling (in one line)
Classification predicts **who will convert**.  
Uplift predicts **who will convert *because* of the treatment**.

---

## Why Uplift Modeling?

In many real-world scenarios (ads, coupons, notifications), applying a treatment to everyone is wasteful or even harmful.

Uplift modeling estimates the **incremental impact of a treatment** by learning:

uplift(x) = P(Y=1 | X=x, T=1) − P(Y=1 | X=x, T=0)

This allows us to:
- Target only users who are **persuadable**
- Avoid spending resources on users who would convert anyway
- Avoid negative impact on users who respond poorly to treatment

In this project, uplift modeling is used to decide **who should receive a treatment**, not just who is likely to convert.

---

## Repository Structure

The repository is organized to clearly separate modeling, deployment, and documentation concerns:

ml-uplift-modeling-criteo/
├── notebooks/                 # Data preparation, modeling, evaluation
│   ├── 01_data_slice.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_uplift_modeling.ipynb
│   ├── 05_evaluation_QINI.ipynb
│   ├── 06_export_model.ipynb
│   └── 07_local_test.ipynb
│
├── lambda_app/                # AWS Lambda inference code
│   └── handler.py
│
├── models/                    # Trained model artifacts
│   └── uplift_tlearner_bundle.joblib
│
├── scripts/                   # Utility scripts
│   ├── build_and_push_ecr.sh
│   └── local_api_test.sh
│
├── documentation/                      # Documentation
│   └── PROJECT_WALKTHROUGH.md
│
├── Dockerfile                 # Lambda container image definition
├── README.md
├── LICENSE
└── .gitignore

---

## Dataset
Using the Criteo uplift dataset (features are anonymized):
- Features: `f0 ... f11`
- Treatment: `treatment` (0/1)
- Outcome: `visit` (0/1)

> Note: The feature meanings are intentionally anonymized by Criteo, but they remain predictive and usable for uplift modeling.

---

## Modeling Approach
### Baseline: T-Learner
Train two models:
- **Treated model** learns: `P(Y=1 | X, T=1)`
- **Control model** learns: `P(Y=1 | X, T=0)`

Inference:
- `p_treated = model_treated.predict_proba(X)`
- `p_control = model_control.predict_proba(X)`
- `uplift = p_treated - p_control`

### Evaluation
Evaluated using uplift metrics (e.g., Qini / AUUC) to measure whether ranking users by predicted uplift improves realized lift.

---

## Deployment Architecture
**Train → Export → Deploy → Serve**
1. Train uplift model locally / in notebooks
2. Export bundle to `models/uplift_tlearner_bundle.joblib`
3. Build Docker image (Lambda Python 3.12 base)
4. Push image to **Amazon ECR**
5. Create **AWS Lambda** from the image
6. Expose endpoint using **API Gateway**: `POST /predict`

---

## Live Endpoint
API Gateway (stage):
- Base: `https://b0t5ntje2g.execute-api.eu-north-1.amazonaws.com/prod`
- Predict: `POST https://b0t5ntje2g.execute-api.eu-north-1.amazonaws.com/prod/predict`

---

## API Contract

### Request
Send a JSON body with `instances` (list of feature dictionaries):

```json
{
  "instances": [
    {
      "f0": 1, "f1": 2, "f2": 3, "f3": 4,
      "f4": 5, "f5": 6, "f6": 7, "f7": 8,
      "f8": 9, "f9": 10, "f10": 11, "f11": 12
    }
  ]
}
```

---

### Response

```json
{
  "predictions": [
    {"p_treated": 0.12, "p_control": 0.08, "uplift": 0.04}
  ],
  "model": "tlearner_logreg_v1",
  "n": 1
}
```

---

## Quick Test CURL

```json
curl -s -X POST "https://b0t5ntje2g.execute-api.eu-north-1.amazonaws.com/prod/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"f0": 1, "f1": 2, "f2": 3, "f3": 4, "f4": 5, "f5": 6, "f6": 7, "f7": 8, "f8": 9, "f9": 10, "f10": 11, "f11": 12}
    ]
  }'
```

---

## API Testing (Postman)

The deployed uplift model API can be tested using Postman.

A Postman collection is included in this repository:
- `documentation/postman_collection.json`

Steps to use:
1. Import the collection into Postman
2. Update the request URL if needed
3. Send a `POST /predict` request with feature values
4. Receive predicted treatment probability, control probability, and uplift

This allows easy validation and experimentation without writing client code.

---

## End-to-End Flow: From Notebook to API

This project follows a complete machine learning lifecycle:

1. **Data Exploration & Preparation**  
   - Raw uplift dataset is explored and cleaned in Jupyter notebooks  
   - Features (`f0`–`f11`), treatment, and outcome variables are identified  

2. **Model Training & Evaluation**  
   - A T-Learner approach is used to train separate models for treated and control groups  
   - Uplift is computed as the difference between predicted probabilities  
   - Model performance is evaluated using uplift-specific metrics (Qini / AUUC)  

3. **Model Export**  
   - Trained models and metadata are bundled and saved using `joblib`  
   - Artifacts are stored in the `models/` directory  

4. **Containerized Deployment**  
   - A Docker image is built using the AWS Lambda Python base image  
   - All dependencies and the trained model are packaged into the container  
   - The image is pushed to Amazon ECR  

5. **Serverless Inference**  
   - AWS Lambda is created from the container image  
   - Memory and timeout are tuned to support sklearn inference  

6. **Public API Exposure**  
   - API Gateway exposes a `POST /predict` endpoint  
   - Requests are validated and batch inference is supported  

7. **API Consumption & Testing**  
   - The deployed API is tested using `curl` and Postman  
   - Predictions return treatment probability, control probability, and uplift  

This flow ensures that the model is not only accurate, but also **deployable, testable, and usable in real-world systems**.

---
