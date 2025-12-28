# Project Walkthrough: Uplift Modeling + AWS Deployment

This document explains the project step by step, using both:
- **Lay Language** (non-technical explanation)
- **Technical Explanation** (engineering and ML details)

It also documents challenges faced during development and the reasoning behind key engineering decisions.

---

## Step 1: What is Uplift Modeling?

### Lay Language
Imagine you run a company and you want to send discount coupons to customers.

If you send coupons to everyone:
- Some customers would buy anyway (you wasted money)
- Some customers will never buy (coupon doesn’t help)
- Some customers buy **only because of the coupon** (these are valuable)
- Some customers may even react negatively

The real question is not:
> “Who will buy?”

The real question is:
> “Who will buy *because* I send the coupon?”

Uplift modeling answers exactly this question.  
It helps decide **who should receive a treatment** so that the action actually makes a positive difference.

### Technical Explanation
Uplift modeling estimates the **conditional average treatment effect (CATE)** for each individual.

Formally, uplift is defined as:

uplift(x) = P(Y = 1 | X = x, T = 1) − P(Y = 1 | X = x, T = 0)

Where:
- `Y` is the outcome (e.g., visit or conversion)
- `T` is the treatment indicator (1 = treated, 0 = control)
- `X` represents user features

Unlike traditional classification models that estimate `P(Y | X)`, uplift models estimate the **difference in outcomes under treatment vs. no treatment**.

This allows decision-making systems to:
- Target only users with positive estimated uplift
- Avoid unnecessary or harmful treatments
- Optimize business impact rather than raw prediction accuracy

---

## Step 2: Business Problem & Why Uplift Matters

### Lay Language
In many businesses, sending offers, ads, or notifications costs money and attention.

If we treat everyone the same way:
- We waste money on users who would act anyway
- We annoy users who dislike being targeted
- We miss the opportunity to focus on users who are actually influenced

Uplift modeling helps answer:
“Who should I target so that my action actually changes behavior?”

Instead of maximizing conversions, the goal becomes maximizing **incremental impact**.

### Technical Explanation
Traditional A/B testing measures average treatment effect across the population, but it does not support **individual-level decision making**.

Uplift modeling allows segment-level or individual-level targeting by estimating treatment effects conditioned on features. This enables:
- selective treatment policies
- cost-aware decision rules
- improved ROI compared to blanket targeting

---

## Step 3: Dataset Understanding

### Lay Language
The dataset contains information about users, whether they received a treatment, and whether they performed an action.

The feature names are anonymized, but they still describe user behavior in numeric form.

Even without knowing what each feature means, the patterns in the data can still be learned.

### Technical Explanation
The dataset contains:
- Features: `f0`–`f11` (anonymized numerical features)
- `treatment`: binary indicator (treated vs control)
- `visit`: binary outcome

Anonymization preserves statistical structure while protecting privacy. The data is suitable for uplift modeling because it includes **both treated and control observations**.

---

## Step 4: Data Cleaning & Preparation

### Lay Language
Before training a model, the data must be cleaned and organized so the model does not learn incorrect patterns.

This includes checking for missing values and making sure treated and control users are properly represented.

### Technical Explanation
Steps performed:
- Verified absence of missing values
- Ensured treatment and control groups were present
- Created stratified train/test splits preserving treatment distribution
- Separated features (`X`), treatment (`T`), and outcome (`Y`)

Proper stratification is critical for unbiased uplift estimation.

---

## Step 5: Baseline Modeling (T-Learner)

### Lay Language
Instead of building one model, we build **two models**:
- One for people who received treatment
- One for people who did not

By comparing their predictions, we can estimate the effect of treatment.

### Technical Explanation
The T-Learner approach trains:
- `model_treated`: predicts `P(Y=1 | X, T=1)`
- `model_control`: predicts `P(Y=1 | X, T=0)`

Uplift is computed as the difference between these predictions.

This approach is simple, interpretable, and serves as a strong baseline.

---

## Step 6: Uplift Modeling

### Lay Language
The model does not just predict outcomes. It predicts **how much the treatment changes the outcome**.

Some users benefit, some don’t, and some may be harmed.

### Technical Explanation
The uplift score is computed as the difference between the predicted outcome probabilities under treatment and control:

uplift(x) = P(Y=1 | X=x, T=1) − P(Y=1 | X=x, T=0)

Using a T-Learner, two independent models are trained:
- A treated model using only samples where `T = 1`
- A control model using only samples where `T = 0`

At inference time, both models are evaluated on the same feature vector.  
The difference between their outputs represents the estimated **individual treatment effect**.

This formulation allows flexible model choices and works well when treated and control groups are sufficiently large.

---

## Step 7: Model Evaluation (Qini / AUUC)

### Lay Language
A normal accuracy score cannot tell us whether the model is good at deciding *who should be treated*.

Instead, we want to know:
“If we start treating users with the highest predicted uplift, do results improve faster than random targeting?”

Qini curves visualize this idea by showing how much extra benefit we gain as we treat more users.

### Technical Explanation
Model evaluation is performed using uplift-specific metrics:
- **Qini Curve**: measures incremental gains from targeting users ranked by predicted uplift
- **AUUC (Area Under the Uplift Curve)**: summarizes ranking performance

The model’s AUUC is compared against:
- random targeting
- baseline uplift approaches

A higher AUUC indicates better prioritization of users with positive treatment effects.

---

## Step 8: Model Export

### Lay Language
Once the model performs well, it is saved so it can be reused without retraining every time.

This allows the model to be deployed and shared.

### Technical Explanation
The trained treated and control models, along with feature ordering metadata, are bundled into a single artifact and saved using `joblib`.

This ensures:
- consistent feature alignment at inference time
- fast loading during deployment
- reproducibility across environments

---

## Step 9: Local Inference Testing

### Lay Language
Before deploying to the cloud, the model is tested locally to ensure it responds correctly to real inputs.

This prevents failures after deployment.

### Technical Explanation
Local tests validate:
- correct parsing of input payloads
- numeric type conversion
- batch inference support
- output schema consistency

Testing locally reduces cloud debugging complexity and speeds up iteration.

---

## Step 10: AWS Deployment (Lambda + Docker)

### Lay Language
To make the model available online, it is placed inside a container and deployed to the cloud.

This allows anyone with the API to request predictions.

### Technical Explanation
Deployment uses:
- Docker container built from the AWS Lambda Python base image
- Amazon ECR to store container images
- AWS Lambda to run inference serverlessly

Container-based deployment avoids Lambda ZIP size limits and ensures binary compatibility with the Linux runtime.

---

## Step 11: API Gateway & Public Endpoint

### Lay Language
An API endpoint is created so applications can send user data and receive predictions automatically.

### Technical Explanation
API Gateway exposes a REST endpoint (`POST /predict`) that forwards requests to Lambda.

The handler performs:
- request validation
- feature extraction
- batch inference
- JSON response formatting

This decouples model logic from clients and enables scalable consumption.

---

## Step 12: API Testing & Validation

### Lay Language
The API is tested to ensure it behaves correctly in both normal and error scenarios.

This builds confidence that the system is reliable.

### Technical Explanation
Tests include:
- valid single and batch requests
- missing or malformed inputs
- empty instance lists
- unexpected data types

The API returns structured error messages with appropriate HTTP status codes.

---

## Challenges & Engineering Decisions

### Problem 1: Lambda ZIP Size Limits

Initial attempts using ZIP-based deployment exceeded Lambda size constraints due to heavy ML dependencies.

**Resolution:** Migrated to Lambda container images.

### Problem 2: OS / Architecture Mismatch

Local builds produced binaries incompatible with Lambda’s Linux runtime.

**Resolution:** Built Docker images targeting `linux/amd64`.

### Problem 3: Lambda Cold Start & Memory Tuning

Low memory allocation caused timeouts during model loading.

**Resolution:** Increased memory allocation to improve CPU and I/O performance.

### Problem 4: Container Image Manifest Issues

Some Docker builds produced image manifests not supported by Lambda.

**Resolution:** Ensured compatible image formats and base images aligned with Lambda requirements.

---

## Final End-to-End Flow

Jupyter Notebook  →  Model Training & Evaluation  →  Model Export  →  Docker Container Build  →  Amazon ECR  →  AWS Lambda  →  API Gateway  →  Postman / Client Applications
