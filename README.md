# ONCOPHOROS

This Python project contains three separate models for testing classification between **HBV-HCC** and **Non-Viral HBV**.

## Project Structure

- **data/**  
  Contains the TCGA-LIHC dataset, split into:

  - `HBV.tsv`
  - `Non-Viral.tsv`

- **lightgbm_version/**  
  Implementation using **LightGBM** (open-source by Microsoft). This is also the main file with most of the comments (as the other programs share the same component). Each of the files can be ran independently and will generate a corresponding folder with the results.

- **logistic_regression/**  
  Implementation using Logistic Regression.

- **random_forest/**  
  Implementation using Random Forest.

- **user_interface/**
  A Simple Gradio user-interface to be able to compare the accuracy of each model.

## Usage

1. Install all required dependencies listed in `requirements.txt`.
2. Update the dataset paths in the code. _(Currently, the paths are set as absolute paths; replace them with the correct paths on your system.)_
