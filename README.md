# MachineLearning

# Bank Churn Prediction Project

This project focuses on predicting customer churn in the banking sector using a dataset of customer attributes and behavior. It employs data preprocessing, traditional machine learning models, and deep learning techniques.

## Project Structure

### 1. Data Preprocessing (`ML_BC_DataPreprocessing.ipynb`)
   - **Purpose**: Cleans and prepares the data for further analysis.
   - **Steps**:
     - Handles missing values and encodes categorical variables.
     - Balances the dataset using oversampling techniques like SMOTE.
     - Normalizes and scales features for better model performance.
   - **How to Run**:
     1. Install required libraries:
        ```bash
        pip install pandas numpy scikit-learn imbalanced-learn
        ```
     2. Download the dataset (details below) and place it in the project directory.
     3. Run the notebook step by step to generate the preprocessed training, testing, and validation datasets.

### 2. Traditional Machine Learning Models (`ML_BC_Traditional_ML.ipynb`)
   - **Purpose**: Implements and evaluates traditional machine learning models.
   - **Models Used**:
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - Support Vector Machines (SVM)
     - Gradient Boosting Machines (GBM)
     - XGBoost
   - **Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - **How to Run**:
     1. Install required libraries:
        ```bash
        pip install pandas numpy scikit-learn matplotlib xgboost
        ```
     2. Load the preprocessed data from the first notebook.
     3. Run the notebook to train and evaluate models on the dataset.

### 3. Deep Learning Models (`ML_BC_DeepLearning.ipynb`)
   - **Purpose**: Applies deep learning models to predict churn.
   - **Techniques Used**:
     - Multi-Layer Perceptrons (MLP)
     - Optimized hyperparameter settings.
     - Cross-validation for performance robustness.
   - **How to Run**:
     1. Install required libraries:
        ```bash
        pip install pandas numpy tensorflow keras
        ```
     2. Load the preprocessed data from the first notebook.
     3. Run the notebook to build, train, and evaluate the deep learning models.

---

## Dataset

- The dataset used in this project is **Credit Card Customers**, available on Kaggle.
- **Download Link**: [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data)
- After downloading the dataset, place the file (`BankChurners.csv`) in the project directory to ensure compatibility with the code.

---

## Execution Order
1. Download and place the dataset in the project directory.
2. Run the **Data Preprocessing** notebook to prepare the data.
3. Execute the **Traditional Machine Learning Models** notebook.
4. Finally, run the **Deep Learning Models** notebook.

---

## Libraries Required
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `matplotlib`
- `tensorflow`
- `keras`
- `xgboost`

---

## Results
Each notebook provides detailed metrics and visualizations for performance comparison across different models. Key insights and best-performing models are highlighted.

---
