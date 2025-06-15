# Mental-Health-Dataset-Analysis-and-Prediction
This repository details a comprehensive machine learning project focused on analyzing and predicting factors related to mental health using a provided dataset. The workflow covers everything from initial data loading and exploration to advanced preprocessing, feature engineering, and the evaluation of various classification models.
## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [1. Mount Google Drive](#1-mount-google-drive)
  - [2. Load Dataset](#2-load-dataset)
  - [3. Data Preprocessing & Consistency](#3-data-preprocessing--consistency)
  - [4. Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
  - [5. Feature Engineering](#5-feature-engineering)
  - [6. Model Training and Evaluation](#6-model-training-and-evaluation)
  - [7. Key Considerations for Model Selection](#7-key-considerations-for-model-selection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

My goal with this project is to explore a mental health dataset, uncover insights, and build predictive models. The process I followed includes:

1.  **Data Loading**: Importing the dataset, specifically the "Mental Health Dataset.csv" file.
2.  **Data Preprocessing**: Handling missing values, encoding categorical variables, scaling numerical features, and performing one-hot encoding for modeling readiness.
3.  **Exploratory Data Analysis (EDA)**: Gaining insights into data distributions, relationships between features, and identifying unusual values.
4.  **Feature Engineering**: Creating new features to enhance model performance.
5.  **Model Training**: Applying and evaluating classification models like Logistic Regression and Random Forest.
6.  **Model Evaluation**: Using key metrics (accuracy, precision, recall, F1-score) and visualizing their performance.

---

## Dataset

The dataset used is `Mental Health Dataset.csv`. Please ensure this file is located in your Google Drive at the path `/content/drive/MyDrive/Mental Health Dataset/` or update the `file_path` accordingly.

---

## Prerequisites

To run this code, you'll need:

-   Python 3.x
-   Google Colab (recommended for seamless Google Drive integration) or a local Python environment.
-   The following Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.

---

## Setup and Installation

1.  **Clone this repository (optional, if you're not using Colab directly):**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install the necessary libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    ```
    If using Google Colab, most of these are pre-installed.

---

## Usage

Follow these steps to execute the analysis and modeling pipeline.

### 1. Mount Google Drive

This step is essential if you're running the code in Google Colab to access your dataset.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Load Dataset
I'm loading the dataset from a specific path in my Google Drive. Remember to update the file_path if your dataset is located elsewhere.

```python

import pandas as pd
try:
    df = pd.read_csv('/content/drive/MyDrive/Mental Health Dataset/Mental Health Dataset.csv') #Change path as needed
    print("Data loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: 'Mental Health Dataset.csv' not found. Make sure the file is in the specified directory or provide the correct path.")
except pd.errors.EmptyDataError:
    print("Error: 'Mental Health Dataset.csv' is empty.")
except pd.errors.ParserError:
    print("Error: Unable to parse the CSV file. Please check the file format.")
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")

```
## 3. Data Preprocessing & Consistency
I apply several preprocessing techniques to ensure the data is clean and consistent for modeling. This includes handling missing values, encoding categorical features, and scaling numerical ones.

```python

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification # Used for synthetic data example

# Handle missing values (replace with mode for categorical, median for numerical)
for col in df.columns:
    if df[col].isnull().any(): # Check if there are any nulls in the column
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# Encode categorical features using Label Encoding (for specific columns like 'Gender' which might be directly used or for general consistency)
label_encoder = LabelEncoder()
# Identify all object-type columns that are still categorical after initial handling
categorical_cols_for_label_encoding = df.select_dtypes(include=['object']).columns
for col in categorical_cols_for_label_encoding:
    df[col] = label_encoder.fit_transform(df[col])

# One-hot encode specified categorical columns for modeling (after initial label encoding for some)
# This list should contain columns you want to convert to dummy variables
categorical_cols_one_hot = ['no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options',
                            'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                            'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview',
                            'phys_health_interview', 'mental_vs_physical']

existing_cols_for_one_hot = [col for col in categorical_cols_one_hot if col in df.columns]
df = pd.get_dummies(df, columns=existing_cols_for_one_hot, drop_first=True)

# Scale numerical features using StandardScaler
# Ensure these are the actual numerical columns present after any encoding
numerical_cols_for_scaling = df.select_dtypes(include=np.number).columns
# Exclude any newly created one-hot encoded columns if they aren't meant for scaling or if target
# For example, if your target is in numerical_cols_for_scaling, you might exclude it.
# For simplicity, I'm scaling all detected numerical columns here.
scaler = StandardScaler()
df[numerical_cols_for_scaling] = scaler.fit_transform(df[numerical_cols_for_scaling])


# Applying Min-Max Scaling on specific features (if desired, as an alternative or complement to StandardScaler)
# Example for numerical features, assuming they exist after other preprocessing
numerical_features_to_minmax_scale = ['Days_Indoors', 'Growing_Stress', 'Changes_Habits'] # Adjust as per your data
minmax_scaler = MinMaxScaler()
for col in numerical_features_to_minmax_scale:
    if col in df.columns:
        # Ensure column is numeric before scaling
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop NaNs that might result from coercion before scaling
        df[col].fillna(df[col].median(), inplace=True) # Or mean, or drop row
        df[col + '_minmax_scaled'] = minmax_scaler.fit_transform(df[[col]])
    else:
        print(f"Warning: column '{col}' not found for Min-Max Scaling.")

# Synthetic Data Handling: If you have missing rows/columns and want to fill with synthetic data,
# this is where you'd integrate it, likely by generating synthetic data and then merging/concatenating
# or using it as a fallback if a column is entirely missing from your original dataset.
# For example, if a specific critical column is missing, you could generate it synthetically.
# Given the current flow, it's more about handling existing missing values robustly.

print("\nDataFrame head after all preprocessing:")
print(df.head())
print("\nDataFrame dtypes after all preprocessing:")
print(df.dtypes)
4. Exploratory Data Analysis (EDA)
I conduct various visualizations and statistical analyses to understand the dataset's characteristics.

Python

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # For Z-score analysis

# --- Categorical Features Analysis ---
# Re-define categorical_features as they might be different after initial preprocessing
# Use columns that were initially categorical but now are label-encoded or one-hot
# You might need to inspect `df.dtypes` after the preprocessing cell to confirm these.
# For simplicity, using a generic check for original categorical columns that are now numerical (due to label encoding)
# Or, if you want to plot the newly created one-hot encoded columns, list them here.
# For the purpose of demonstration, let's assume 'Gender', 'Country', 'self_employed', 'treatment'
# were label-encoded and we want to see their distributions.
# Note: For one-hot encoded columns, countplot directly on the original column name (before encoding) is better.
# Or, if you want to see counts of 'no_employees_1.0', 'no_employees_2.0' etc., you would list them.

# Re-evaluate which columns are suitable for these plots *after* your preprocessing.
# If 'Gender' is now 'Gender_encoded' from LabelEncoder, use that.
# For one-hot encoded columns like 'no_employees_X', you'd look at original 'no_employees'.

# Example for original categorical columns (assuming they are now numerical due to LabelEncoder)
# You might need to adjust column names based on your actual data after preprocessing
original_categorical_display = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment']
for col in original_categorical_display:
    # Check if the original column name exists or if its encoded version exists
    if col in df.columns: # If the original column was label encoded and still exists
        print(f"\n--- Distribution of {col} (after encoding if applicable) ---")
        print(df[col].value_counts())
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, data=df, palette='viridis')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    elif f"{col}_encoded" in df.columns: # If you created an explicit encoded column
        print(f"\n--- Distribution of {col}_encoded ---")
        print(df[f"{col}_encoded"].value_counts())
        plt.figure(figsize=(10, 6))
        sns.countplot(x=f"{col}_encoded", data=df, palette='viridis')
        plt.title(f'Distribution of {col}_encoded')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print(f"\n--- Column '{col}' or its encoded version not found for plotting categorical distribution ---")


# --- Numerical Features Analysis ---
numerical_features = ['Days_Indoors', 'Growing_Stress', 'Changes_Habits', 'Mood_Swings',
                      'Coping_Struggles', 'Work_Interest', 'Social_Weakness'] # Add other numerical columns as needed

for col in numerical_features:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        print(f"\n--- {col} ---")
        print(df[col].describe())

        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()
    else:
        print(f"\n--- Column '{col}' not found or not numerical in DataFrame ---")

# --- Scatter Plots for Relationships ---
if 'Days_Indoors' in df.columns and 'Growing_Stress' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Days_Indoors', y='Growing_Stress', data=df)
    plt.title('Scatter Plot: Days_Indoors vs. Growing_Stress')
    plt.xlabel('Days_Indoors')
    plt.ylabel('Growing_Stress')
    plt.show()
else:
    print("One or both of the specified columns ('Days_Indoors', 'Growing_Stress') were not found for scatter plot.")

# --- Boxplot: Days_Indoors by Gender (assuming Gender is encoded or original) ---
if 'Days_Indoors' in df.columns and 'Gender' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gender', y='Days_Indoors', data=df)
    plt.title('Boxplot: Days_Indoors by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Days_Indoors')
    plt.show()
else:
    print("One or both of the specified columns ('Days_Indoors', 'Gender') were not found for boxplot.")

# --- Violin Plot: Days_Indoors by Gender ---
if 'Days_Indoors' in df.columns and 'Gender' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Gender', y='Days_Indoors', data=df)
    plt.title('Violin Plot: Days_Indoors by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Days_Indoors')
    plt.show()
else:
    print("One or both of the specified columns ('Days_Indoors', 'Gender') were not found for violin plot.")

# --- Cross-tabulations and Heatmaps for Categorical Correlation ---
def analyze_categorical_correlation(df, col1, col2):
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: One or both columns ('{col1}', '{col2}') not found in the DataFrame.")
        return
    cross_tab = pd.crosstab(df[col1], df[col2], normalize='index') * 100
    print(f"\n--- Cross-tabulation: {col1} vs. {col2} ---")
    print(cross_tab)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap='viridis', linewidths=.5)
    plt.title(f'Correlation Heatmap: {col1} vs. {col2}')
    plt.show()

# Example usage (use columns that are now numerical due to encoding or were originally numerical)
# Assuming 'self_employed', 'family_history', 'treatment', 'work_interfere' are now numerical after encoding
analyze_categorical_correlation(df, 'self_employed', 'family_history')
analyze_categorical_correlation(df, 'treatment', 'work_interfere')

# --- Numerical Feature Correlation Matrix ---
numerical_df = df.select_dtypes(include=['number'])
correlation_matrix = numerical_df.corr()

print("\n--- Correlation Matrix of All Numerical Features ---")
print(correlation_matrix)

plt.figure(figsize=(16, 12)) # Adjusted size for potentially more features
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of All Numerical Features')
plt.show()

# --- Unusual Values Detection (Outliers via IQR and Z-score) ---
def detect_unusual_values_iqr(df, column_name):
    if column_name not in df.columns or not pd.api.types.is_numeric_dtype(df[column_name]):
        print(f"Error: Column '{column_name}' not found or not numerical.")
        return None
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    unusual_values = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
    return unusual_values

print("\n--- Outlier Detection (IQR Method) ---")
for col in numerical_features: # Re-using the list of primary numerical features
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        unusual_in_col = detect_unusual_values_iqr(df, col)
        if unusual_in_col is not None and unusual_in_col.any(): # Check if any outliers exist
            print(f"\nUnusual values (IQR) in '{col}':")
            display(df[unusual_in_col][[col]]) # Display only the column with outliers
        else:
            print(f"\nNo significant IQR outliers found in '{col}'.")
    else:
        print(f"\nColumn '{col}' not suitable for IQR outlier detection (missing or not numerical).")

print("\n--- Outlier Detection (Z-score Method) ---")
for col in numerical_features: # Re-using the list of primary numerical features
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        valid_data = df[col].dropna()
        if not valid_data.empty: # Only calculate if there's valid data
            df[col + '_zscore'] = pd.Series(stats.zscore(valid_data), index=valid_data.index)
            print(f"\nZ-scores for {col}:")
            print(df[col + '_zscore'].describe())

            outliers_z = df[(df[col + '_zscore'] > 3) | (df[col + '_zscore'] < -3)]
            if not outliers_z.empty:
                print(f"\nOutliers in {col} (|z-score| > 3):")
                display(outliers_z[[col, col + '_zscore']])
            else:
                print(f"\nNo significant Z-score outliers found in '{col}'.")
        else:
            print(f"\nNo valid data in '{col}' to calculate Z-scores.")
    else:
        print(f"\nColumn '{col}' not suitable for Z-score analysis (missing or not numerical).")

# --- Grouped Bar Plots ---
def grouped_bar_plot(df, col1, col2, col3):
    if not all(c in df.columns for c in [col1, col2, col3]):
        print(f"Error: One or more columns ('{col1}', '{col2}', '{col3}') not found in the DataFrame.")
        return

    plt.figure(figsize=(12, 6))
    sns.countplot(x=col1, hue=col2, data=df, palette='viridis')
    plt.title(f'Grouped Bar Plot: {col1} by {col2}')
    plt.xlabel(col1)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.countplot(x=col1, hue=col3, data=df, palette='viridis') # Using same palette for consistency
    plt.title(f'Grouped Bar Plot: {col1} by {col3}')
    plt.xlabel(col1)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Example usage: Replace 'Occupation' with an actual column name that would be suitable
# Assuming 'Gender', 'family_history', and 'mental_health_interview' are suitable for this plot
grouped_bar_plot(df, 'Gender', 'family_history', 'mental_health_interview')

# --- Analysis of Mental Health Features ---
print("\n--- Analysis: Family History, Treatment, and Mental Health Interview ---")
analyze_categorical_correlation(df, 'family_history', 'treatment')
analyze_categorical_correlation(df, 'treatment', 'mental_health_interview')
analyze_categorical_correlation(df, 'family_history', 'mental_health_interview')

print("\n--- Analysis: Days Indoors vs. Work Interest/Social Weakness ---")
if 'Days_Indoors' in df.columns and 'Work_Interest' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Days_Indoors', y='Work_Interest', data=df)
    plt.title('Scatter Plot: Days_Indoors vs. Work_Interest')
    plt.show()
else:
    print("Columns 'Days_Indoors' or 'Work_Interest' not found for scatter plot.")

if 'Days_Indoors' in df.columns and 'Social_Weakness' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Days_Indoors', y='Social_Weakness', data=df)
    plt.title('Scatter Plot: Days_Indoors vs. Social_Weakness')
    plt.show()
else:
    print("Columns 'Days_Indoors' or 'Social_Weakness' not found for scatter plot.")

print("\n--- Analysis: Care Options by Country/Gender ---")
# Ensure 'Country' and 'care_options' are suitable for this analysis after encoding
# If 'Country' is now one-hot encoded, you might need to use original 'Country' before one-hot or look at specific dummy columns.
# For simplicity, assuming 'Country' and 'care_options' are still usable as single columns (e.g., if label encoded)
if 'Country' in df.columns and 'care_options' in df.columns and 'Gender' in df.columns:
    grouped_bar_plot(df, 'Country', 'care_options', 'Gender')
else:
    print("Columns 'Country', 'care_options', or 'Gender' not found for grouped bar plot.")

# Additional analysis using boxplots (Care Options by Country/Gender)
# These plots require numerical 'y' axis. If 'care_options' or 'Country' are categorical, this might not directly apply.
# You could use a count plot or another aggregated view.
# The original code uses df.index for the y-axis, which is a workaround. Better to use a meaningful numerical feature.
# Let's adjust to be more robust for categorical features or use relevant numerical features.
# For example, if you want to see if 'care_options' (after encoding) varies with numerical 'Age' if you had it.
# Or just use the original categorical plot methods above.
# Skipping the direct boxplots with df.index for this README as it's less interpretable.
```
## 5. Feature Engineering
I'm creating new features from existing ones to potentially improve model performance.

```python

import numpy as np

# Example feature engineering steps (add more as needed)

# 1. Create an interaction feature (e.g., product of two numerical features)
if 'Days_Indoors' in df.columns and 'Growing_Stress' in df.columns:
    df['Days_Stress_Interaction'] = df['Days_Indoors'] * df['Growing_Stress']
else:
    print("One or both columns 'Days_Indoors' or 'Growing_Stress' are not in the DataFrame for interaction feature.")

# 2. Create a binary feature based on a threshold (example with 'Days_Indoors')
# Ensure 'Days_Indoors' is numerical
if 'Days_Indoors' in df.columns and pd.api.types.is_numeric_dtype(df['Days_Indoors']):
    df['High_Days_Indoors'] = (df['Days_Indoors'] > df['Days_Indoors'].median()).astype(int) # Using median as threshold
else:
    print("Column 'Days_Indoors' not found or not numerical for binary feature.")

# 3. Create a new feature by grouping values (example with 'Country')
# This assumes 'Country' is still in its original string format or has been processed appropriately.
# If 'Country' was label-encoded, this might need re-evaluation.
# For one-hot encoded 'Country_USA', 'Country_Canada' etc., this becomes more complex.
# This example works best if 'Country' is still a string column.
if 'Country' in df.columns and isinstance(df['Country'].iloc[0], str): # Check if it's string-like
    country_groups = {
        'United States': 'North America',
        'Canada': 'North America',
        'United Kingdom': 'Europe',
        'Australia': 'Oceania',
        'India': 'Asia',
        # ...add other countries and groups...
    }
    df['Country_Group'] = df['Country'].map(country_groups).fillna('Other')
    # If Country_Group is created, it might need encoding later if used as a feature.
    # For now, it's a new string column.
else:
    print("Column 'Country' is not in the DataFrame or not suitable for direct grouping (might be encoded already).")


# 4. Log transformation for skewed numerical data
# Ensure 'Days_Indoors' is numerical and consider its distribution
if 'Days_Indoors' in df.columns and pd.api.types.is_numeric_dtype(df['Days_Indoors']):
    # Add a small constant if values can be zero, to avoid log(0)
    df['Log_Days_Indoors'] = np.log1p(df['Days_Indoors'])
else:
    print("'Days_Indoors' column not found or not numerical for log transformation.")


print("\nDataFrame head after feature engineering:")
print(df.head())
```
## 6. Model Training and Evaluation
Here, I prepare the data for modeling, train Logistic Regression and Random Forest classifiers, and evaluate their performance.

```python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Re-run one-hot encoding if needed after feature engineering
# (Ensuring any new categorical features from feature engineering are also handled)
# If Country_Group was created and is still string, it needs to be one-hot encoded or label encoded.
# For simplicity, assuming the 'df' passed to this section is ready for modeling (all features are numeric).

# Identify a target variable. **YOU MUST REPLACE 'target_variable' with your actual target column name.**
# For a mental health dataset, this could be 'treatment', 'mental_health_consequence', or similar.
# Make sure your target variable is binary or multi-class for classification.
# For example, if 'treatment' is your target and it's already encoded to 0/1, use that.
# Let's assume 'treatment' is the target variable which has been encoded to numeric values.
target_variable_name = 'treatment' # <--- *** IMPORTANT: REPLACE THIS WITH YOUR ACTUAL TARGET COLUMN NAME ***

# Drop columns that are not suitable as features (e.g., original string columns after encoding, or ID columns)
# Ensure you don't drop your target variable!
columns_to_drop_from_X = ['Country_Group'] # Example, adjust based on your feature engineering and what you want to include
X = df.drop(columns=[col for col in columns_to_drop_from_X if col in df.columns] + [target_variable_name], axis=1, errors='ignore')
y = df[target_variable_name]

# Ensure X contains only numerical data
X = X.select_dtypes(include=np.number)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Use stratify for imbalanced classes

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42), # Increased max_iter for convergence
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}
y_preds = {} # Store predictions for graphical comparison

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_preds[name] = y_pred # Store predictions

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=1),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=1)
    }

# Print the results
for name, metrics in results.items():
    print(f"--- {name} Performance ---")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# --- Plotting Metrics Comparison ---
metrics_names = ['Precision', 'Recall', 'F1-score']
# Extract values from results dictionary
precision_values = [results[model_name]['precision'] for model_name in models.keys()]
recall_values = [results[model_name]['recall'] for model_name in models.keys()]
f1_values = [results[model_name]['f1'] for model_name in models.keys()]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precision_values, width, label='Precision')
rects2 = ax.bar(x, recall_values, width, label='Recall')
rects3 = ax.bar(x + width, f1_values, width, label='F1-score')

ax.set_ylabel('Scores')
ax.set_title('Precision, Recall, and F1-score Comparison by Model')
ax.set_xticks(x)
ax.set_xticklabels(models.keys())
ax.legend()
plt.ylim(0.0, 1.0) # Ensure y-axis shows full range of scores
plt.tight_layout()
plt.show()

# --- Plotting Accuracy Comparison ---
accuracy_values = [results[model_name]['accuracy'] for model_name in models.keys()]

plt.figure(figsize=(8, 6))
sns.barplot(x=list(models.keys()), y=accuracy_values, palette='coolwarm')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classification Models')
plt.ylim(0.0, 1.0)
plt.show()
```
## 7. Key Considerations for Model Selection
When choosing the best model, I consider:
```python
Performance Metrics: For classification, I focus on Accuracy, Precision, Recall, and F1-score. For regression (if applicable), I'd use MAE, MSE, and R².
Comparative Analysis: I directly compare models based on these metrics to see which performs best for my specific task.
Trade-offs: I assess the balance between a model's simplicity (ease of understanding), interpretability (how well I can explain its decisions), and its raw performance. Sometimes a slightly less accurate but more interpretable model is preferred depending on the project's goals.
Results
After running the models, the performance metrics will be printed and visualized. An example of expected output might be:

Logistic Regression Performance:

Accuracy: [e.g., 0.8850]
Precision: [e.g., 0.8845]
Recall: [e.g., 0.8850]
F1-score: [e.g., 0.8847]
Random Forest Performance:

Accuracy: [e.g., 0.9120]
Precision: [e.g., 0.9118]
Recall: [e.g., 0.9120]
F1-score: [e.g., 0.9119]
(Actual values will vary based on your dataset and random splits.)
```

The generated bar graphs will visually represent these scores, making it easy to compare Logistic Regression and Random Forest directly on Precision, Recall, F1-score, and Accuracy.
