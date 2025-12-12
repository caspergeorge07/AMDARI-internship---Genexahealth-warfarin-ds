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