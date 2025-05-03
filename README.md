# Personal Loan Campaign - Machine Learning Project

## Overview
This project aims to build a machine learning model to predict the likelihood of customers accepting a personal loan offer for AllLife Bank. By analyzing customer demographics, financial behavior, previous campaign responses, and bank product usage, the insights derived from this model can help in targeting the right customers and optimizing the marketing campaign strategy.

## Objectives
- Identify customers likely to accept a personal loan.
- Perform exploratory data analysis (EDA) to understand customer behavior.
- Pre-process the data to prepare it for modeling.
- Build and evaluate a predictive model.
- Provide actionable insights for the marketing team.

## Data Source
The dataset used for this project is located in the `data` directory:
- `data/Loan_Modelling.csv`: This file contains the dataset, including customer demographics, financial behavior, previous campaign responses, and bank product usage.

## Project Structure
- `notebooks/loan_project.ipynb`: The main Jupyter notebook for the project, containing sections for problem definition, EDA, data pre-processing, model building, performance evaluation, and actionable insights.
- `src/data_processing.py`: Contains functions for data pre-processing tasks such as handling missing values, detecting and treating outliers, and feature engineering.
- `src/model.py`: Implements the decision tree model, defines evaluation criteria, builds the model, visualizes decision rules, and assesses feature importance.
- `src/visualization.py`: Contains functions for creating visualizations for EDA, including univariate and bivariate analyses, as well as correlation heatmaps.
- `requirements.txt`: Lists the required Python packages and their versions needed to run the project.

## Instructions
1. Clone the repository or download the project files.
2. Install the required packages listed in `requirements.txt` using pip:
   ```
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebook `notebooks/loan_project.ipynb` to start the analysis.
4. Follow the sections in the notebook for EDA, data pre-processing, model building, and evaluation.

## Conclusion
This project will provide valuable insights into customer behavior regarding personal loan acceptance, enabling AllLife Bank to enhance its marketing strategies and improve customer targeting.