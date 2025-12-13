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


👤 Author

Ndubuaku Casper Ekwueme
Data Scientist Intern
GenexaHealth Data Project