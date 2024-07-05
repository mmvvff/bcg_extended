# Problem Statement

In today's competitive utilities landscape, a robust defensive marketing strategy focused on minimizing customer churn is crucial for maintaining a loyal customer base. This data science project aims to tackle this issue by providing valuable insights to inform customer churn prevention programs at PowerCo. We will develop a predictive model for customer churn in the utilities sector, focusing on identifying high-risk customers and understanding the key factors contributing to churn.

**Hypothesis:**
Our primary hypothesis is that price factors are the main drivers of customer churn in the utilities sector. Specifically, we posit that:

1. Price sensitivity is the strongest predictor of churn probability among all variables considered.
2. Non-price factors, while potentially influential, play a secondary role in determining customer churn.

Through our analysis, we examine this hypothesis and explore the relative importance of price and non-price factors in predicting customer churn.

## Data requirements
Ideally, we require historical data on customer characteristics (e.g., billing rates, service usage patterns, customer complaints, and a churn indicator), and on service characteristics (e.g., type of energy, service failures and pricing information).

## Project Structure

### [Notebook 1: Data Preparation and Exploration](https://github.com/mmvvff/bcg_extended/blob/main/baseline_notebooks/01_eda.ipynb)

1.1. Data Preprocessing and Cleaning
- Handle missing values
- Remove duplicates
- Address consistency and data formatting.

1.2. Exploratory Data Analysis
- Visualize distributions of key variables
- Analyze correlations between features
- Identify potential patterns or trends related to churn

### [Notebook 2: Feature Engineering](https://github.com/mmvvff/bcg_extended/blob/main/baseline_notebooks/02_feature_engineering.ipynb)

2.1. Create Relevant Variables: Derive new features from existing data

2.2. Encode Categorical Variables: Convert categorical data into numerical format

2.3. Feature Selection: Identify and select the most important features for the model

### [Notebook 3: Modeling and Interpretation](https://github.com/mmvvff/bcg_extended/blob/main/baseline_notebooks/03_modeling_RF_final.ipynb)

3.1. Model Training and Evaluation
- Train multiple variations of Random Forest models
- Evaluate model performance using appropriate metrics
- Select the best-performing model variation

3.2. Model Interpretation and business insights
- Identify key factors contributing to churn
- Compare importance of price vs. non-price factors
- Examine relationships between key factors and customer churn probability

3.3. Insights and Recommendations
- Summarize key findings from the model
- Provide actionable recommendations
- Discuss model limitations and areas for future research

## Expected Outcomes:
1. A predictive model capable of identifying customers at high risk of churning
2. Insights into the main factors driving customer churn
3. Actionable recommendations for improving customer retention strategies