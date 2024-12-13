# Bank Churn Prediction Project

## Project Overview
This project focuses on predicting credit card customer churn using advanced data preprocessing, feature selection, and a variety of machine learning and deep learning models. Customer churn prediction helps financial institutions identify customers who are likely to discontinue their services, enabling targeted retention strategies.

### Key Objectives:
- Determine whether there is a significant difference in average credit utilization ratio between male and female card holders.
- Achieve highly accurate churn predictions through data cleaning, class imbalance handling (via SMOTE), feature selection, and model tuning.
- By running through the notebooks and scripts provided, you will be able to preprocess the dataset, build various ML models (e.g., Logistic Regression, Random Forest, XGBoost), and train deep learning models (MLP, TabTransformer) to compare their performance.

## Data and Files

### Dataset:
- The dataset `BankChurners.csv` (or similarly named CSV) is publicly available from Kaggle.
- **Download Link**: [Credit Card Customers Dataset](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data)

- What it contains:
  - Customer demographic details (e.g., Age, Gender, Marital Status, Income Category)
  - Financial attributes (e.g., Credit Limit, Avg Utilization Ratio, Total Transaction Amount)
  - Churn Label (`Attrition_Flag`) indicating if the customer has left or remained.
- Please place the `BankChurners.csv` file in the same directory as the notebooks so it can be loaded without additional path modifications.

### Notebooks:

#### `ML_BC_Visualization.ipynb`
- Run this first to understand the data distribution, visualize features, and examine initial insights (like the distribution of `Attrition_Flag`, Age, Education, Income, and Gender differences).
- This notebook also addresses the research question on the difference in average credit utilization ratio between male and female customers.

#### `ML_BC_DataPreprocessing.ipynb`
- Run this second to clean and preprocess the data.
- Steps include:
  - Dropping irrelevant columns.
  - Splitting into train, validation, and test sets.
  - Handling placeholder values (e.g., "Unknown") and imputing missing categories.
  - Encoding categorical variables, scaling numerical features.
  - Applying SMOTE to address class imbalance.
- Outputs: `balanced_train_data.csv`, `validation_data.csv`, and `test_data.csv` are generated after preprocessing.

#### `ML_BC_Traditional_ML.ipynb`
- Run this third.
- Uses the processed CSV files (`balanced_train_data.csv`, `validation_data.csv`, `test_data.csv`) to train various traditional and ensemble ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
- Performs feature selection (RFE, Random Forest importance) and hyperparameter tuning (especially for XGBoost).
- Evaluates and compares model performance using accuracy, precision, recall, F1-score, and ROC AUC.

#### `ML_BC_DeepLearning.ipynb`
- Run this last.
- Trains deep learning models (MLP, TabTransformer) on the same processed CSV files.
- Compares results with the best ML models from the previous notebook.
- Concludes with insights on why deep learning might not outperform ensemble methods given the dataset size.

### Scripts:
- If any Python scripts are provided (e.g., `utils.py`, `models.py`), they can be referenced within the notebooks. The notebooks are designed to be self-contained, but scripts can help modularize code.


## Installation and Usage

### Python Version:
- Python 3.x is recommended.

### Required Packages (Install via pip):
```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install imbalanced-learn
pip install torch
pip install xgboost
```

### Optional Packages (If not already installed):
```bash
pip install jupyter
pip install ipython
```

## Running the Notebooks:
- Ensure BankChurners.csv is in the same directory as the notebooks.
- Launch Jupyter Notebook or JupyterLab:
```bash
  jupyter notebook
```
  or
```bash
  jupyter lab
```
  
- Open ML_BC_Visualization.ipynb and run all cells to visualize data and perform initial EDA.
- Open ML_BC_DataPreprocessing.ipynb and run all cells to preprocess data and generate the balanced_train_data.csv, validation_data.csv, and test_data.csv.
- Open ML_BC_Traditional_ML.ipynb to train, tune, and evaluate traditional and ensemble ML models.
- Finally, open ML_BC_DeepLearning.ipynb to train and evaluate deep learning models.

## Note:
- The training process (especially with XGBoost and deep learning models) may take some time depending on your system’s hardware.
- Ensure that you have enough memory and CPU/GPU resources if you plan on extensive hyperparameter tuning or running deep learning models efficiently.
 
## Results and Findings
- The analysis shows that female customers have a statistically higher average credit utilization ratio compared to male customers.
- After preprocessing, SMOTE balancing, feature selection, and tuning, XGBoost achieved the best churn prediction performance (~96.6% accuracy and high AUC).
- Deep learning models, while competitive, did not surpass the best-tuned ensemble methods—likely due to the dataset’s relatively modest size.
- The study confirms the importance of proper data preprocessing, handling class imbalance, and hyperparameter tuning in achieving state-of-the-art performance in churn prediction.
  
## Future Work
- Experiment with larger datasets or additional features to potentially improve deep learning results.
- Incorporate explainability tools (e.g., SHAP) to understand model decision-making.
- Explore cost-sensitive learning to prioritize predictions for high-value customers.
  
## License
- This project and code are for educational and research purposes.
- Consult the dataset’s original source (Kaggle) for any usage restrictions or conditions.

