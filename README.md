# GenexaHealth – Warfarin Data Science

This repository contains data engineering and data science workflows for the
GenexaHealth Warfarin project.

## Structure
- `src/data/` – API fetching, validation, and merging scripts
- `data/raw/` – Raw data pulled from the API
- `data/curated/` – Cleaned and merged datasets
- `docs/` – Data documentation

## Workflow
1. Authenticate with the API
2. Fetch data per table (genomics, clinical, lifestyle, outcomes)
3. Validate schemas
4. Merge into analysis-ready datasets

This repo is designed to support downstream analytics and modeling.

🧬 GenexaHealth Data Pipeline

This repository contains a reproducible data ingestion and merging pipeline for the GenexaHealth API.
It fetches multiple healthcare-related datasets, stores them in a structured format, and merges them into a single analytical dataset.

📌 Project Overview
The pipeline performs the following steps:
  - Authenticates with the GenexaHealth API
  - Fetches paginated datasets from multiple endpoints
  - Saves raw datasets to disk (unchanged)
  - Merges all datasets into a single, patient-level dataset
  - Pushes versioned data and scripts to GitHub

This design follows data engineering best practices:
  - Raw vs processed data separation
  - Reproducible scripts
  - Clear merge logic
  - Git-based version control

📂 Repository Structure
genexahealth-data/
│
├── data/
│   ├── raw/                     # Raw API extracts (no transformations)
│   │   ├── patient_ids.csv
│   │   ├── genomics.csv
│   │   ├── clinical.csv
│   │   ├── lifestyle.csv
│   │   └── outcomes.csv
│   │
│   └── processed/               # Processed / merged outputs
│       └── merged_patient_dataset.csv
│
├── scripts/
│   ├── fetch_table.py           # Generic paginated fetch script
│   ├── fetch_patient_ids.py     # Fetch patient IDs
│   └── merge_all.py             # Merge all datasets
│
├── .env                         # API credentials (not committed)
├── .gitignore
└── README.md

📊 Datasets
Raw datasets (data/raw/)
File	              Description
patient_ids.csv    	Master list of patient identifiers
genomics.csv	      Genomic markers per patient
clinical.csv	      Clinical records
lifestyle.csv	      Lifestyle information
outcomes.csv	      Health outcomes

Processed dataset (data/processed/)
File                      	Description
merged_patient_dataset.csv	Patient-level dataset created by joining all raw tables


🔑 Environment Setup
Create a .env file in the project root:
BASE_URL=https://genexahealth.onrender.com/api/v1
ACCESS_TOKEN=your_access_token_here

 ⚠️ The .env file is ignored by Git and must not be committed.


▶️ How to Run the Pipeline
1️⃣ Fetch raw datasets
Fetch paginated datasets (reuse the same script):

python scripts/fetch_table.py genomics data/raw/genomics.csv
python scripts/fetch_table.py clinical data/raw/clinical.csv
python scripts/fetch_table.py lifestyle data/raw/lifestyle.csv
python scripts/fetch_table.py outcomes data/raw/outcomes.csv

Fetch patient IDs:
python scripts/fetch_patient_ids.py


2️⃣ Merge datasets
Create the unified patient-level dataset:
python scripts/merge_all.py

Output:
data/processed/merged_patient_dataset.csv

🔗 Merge Logic
  - All datasets are merged on patient_id
  - Column name casing differences (e.g. Patient_ID vs patient_id) are handled automatically
  - A left join strategy is used to ensure all patients are retained even if some records are missing

✅ Quality Checks Performed
  - Pagination verified (50,000+ records fetched)
  - Unique patient ID validation
  - Schema alignment across tables
  - Raw data preserved unchanged
  - Processed data versioned separately

# 🧬 GenexaHealth — Explainable Warfarin Dose Recommendation System

## 📌 Project Overview

This project delivers an **end-to-end, explainable clinical decision-support prototype** for predicting a patient’s **Final Stable Warfarin Dose (mg)** using clinical, demographic, and genetic data.

The system combines:
- Robust **machine learning modelling** (XGBoost)
- **Experiment tracking** (MLflow)
- **Global & patient-level explainability** (SHAP & LIME)
- A **clinician-friendly prototype application** (Gradio)

⚠️ **Important:**  
This system is designed strictly as a **clinical decision-support tool**, not an automated prescribing system. Final dosing decisions must remain with qualified healthcare professionals.

---

## 🎯 Problem Statement

Warfarin dosing is clinically challenging due to:
- High inter-patient variability
- Strong genetic influences (e.g., *CYP2C9*, *VKORC1*)
- Risk of serious adverse events from under- or over-dosing

The goal of this project is to:

> **Predict a patient’s final stable warfarin dose accurately, transparently, and in a clinically interpretable manner.**

---

## 🧠 Solution Summary

The system:
1. Ingests validated clinical and genetic patient data
2. Applies feature engineering informed by pharmacogenomic evidence
3. Trains and evaluates multiple ML models
4. Selects the best-performing and most stable model
5. Explains predictions at both global and patient levels
6. Exposes results through an interactive prototype application

---

## 🗂 Project Structure

genexahealth-data/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── Day1_Data_Ingestion.ipynb
│ ├── Day2_EDA_Feature_Engineering.ipynb
│ ├── Day3_Baseline_Models.ipynb
│ ├── Day4_Advanced_Modeling_MLflow.ipynb
│ └── Day5_Explainability_Gradio.ipynb
│
├── artifacts/
│ └── day5_explainability/
│ ├── shap_global_summary.png
│ ├── shap_patient_waterfall.png
│ ├── lime_patient.html
│
├── docs/
│ ├── GenexaHealth_Full_Detailed_Report.pdf
│ └── GenexaHealth_Full_Detailed_Report.docx
│
├── app/
│ └── gradio_app.py
│
└── mlflow/
└── README.md

# 📊 Data & Feature Engineering

### Input Data
- Demographics: Age, Weight, Height, Ethnicity
- Genetics: *CYP2C9*, *VKORC1*
- Clinical factors: Concomitant medications (e.g., Amiodarone)

### Feature Engineering Highlights
- Pharmacogenomic risk scores derived from genotypes
- BMI derived from height & weight
- One-hot encoding for categorical variables
- Strict separation between **raw clinical inputs** and **engineered model features**

This ensures:
- Model performance
- Interpretability
- Production safety

---

## 🤖 Modeling Approach

### Models Evaluated
- Linear Regression (Ridge)
- Random Forest
- Neural Network (MLP)
- **XGBoost (Final Model)**

### Model Selection Criteria
- Predictive accuracy
- Stability across validation
- Explainability suitability
- Clinical plausibility

---

## 📈 Model Performance (XGBoost)

| Metric | Result | Interpretation |
|------|------|---------------|
| RMSE | ~0.54 mg | Low average prediction error |
| MAE | ~0.39 mg | Clinically acceptable |
| R² | ~0.88 | Strong explanatory power |
| Within ±20% | ~82% | High clinical agreement |
| Within ±5 mg | 100% | Safety tolerance met |

---

## 🔍 Explainability & Transparency

### Global Explainability (SHAP)
- Identifies the most influential features across all patients
- Top drivers align with established medical literature:
  - *CYP2C9* metabolic risk
  - *VKORC1* sensitivity
  - Age
  - Amiodarone use
  - Ethnicity

### Patient-Level Explainability
- SHAP waterfall plots explain **why a specific dose was recommended**
- LIME provides local surrogate explanations as a secondary method

This ensures:
- Clinical trust
- Auditability
- Regulatory transparency

---

## 🧪 Experiment Tracking (MLflow)

MLflow is used for:
- Tracking experiments, parameters, and metrics
- Comparing model versions
- Ensuring reproducibility

📌 **Note:**  
MLflow databases and artifacts are intentionally excluded from GitHub due to size and best-practice considerations.  
See `mlflow/README.md` for instructions to reproduce experiments locally.

---

## 🖥 Prototype Application (Gradio)

### What the Prototype Demonstrates
- Clinician-friendly input form (no CSV uploads)
- Real-time dose prediction
- Automatic SHAP explanation generation
- Safe separation between user inputs and engineered features

### How Stakeholders Can Test
- The prototype can be exposed via a **temporary public Gradio link**
- No Python or local setup required for reviewers

---

## ⚖️ Clinical & Regulatory Considerations

- Decision-support only (no automated prescribing)
- Human oversight required
- Transparent feature contributions
- Reproducible and auditable pipeline
- Explicit documentation of assumptions & limitations

---

## 🚧 Limitations & Future Work

### Current Limitations
- Prototype-level deployment
- No live EHR integration
- Single dataset source
- No prospective clinical validation

### Future Enhancements
- External dataset validation
- Prediction uncertainty intervals
- EHR/API integration
- Role-based access control
- Regulatory model cards

---

## 🛠 How to Run Locally

```bash
git clone <repo-url>
cd genexahealth-data
pip install -r requirements.txt


👤 Author

Ndubuaku Casper Ekwueme
Data Scientist Intern
GenexaHealth Data Project