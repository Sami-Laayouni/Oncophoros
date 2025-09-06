# ONCOPHOROS

This Python project contains three separate models for testing classification between **HBV-HCC** and **Non-Viral HBV**.

## Project Structure

- **data/**  
  Contains the TCGA-LIHC dataset, split into:

  - `HBV.tsv`
  - `Non-Viral.tsv`

- **lightgbm_version/**  
  Implementation using **LightGBM** (open-source by Microsoft).

- **logistic_regression/**  
  Implementation using Logistic Regression.

- **random_forest/**  
  Implementation using Random Forest.

## Usage

1. Install all required dependencies listed in `requirements.txt`.
2. Update the dataset paths in the code. _(Currently, the paths are set as absolute paths; replace them with the correct paths on your system.)_
