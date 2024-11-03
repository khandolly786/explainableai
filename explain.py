import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

def load_data(file_path):
    """
    Load dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("File not found. Please check the file path.")
        return None

def preprocess_data(data):
    """
    Perform data preprocessing, such as handling missing values and encoding categorical variables.
    
    Parameters:
        data (DataFrame): Input data to preprocess.
    
    Returns:
        DataFrame: Preprocessed data.
    """
    # Drop rows with missing values
    data = data.dropna()
    logging.info("Missing values dropped.")

    # Example of encoding categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols)
    logging.info("Categorical columns encoded.")
    return data

def split_data(data, target_column):
    """
    Split the data into training and testing sets.
    
    Parameters:
        data (DataFrame): Preprocessed data.
        target_column (str): Name of the target column.
    
    Returns:
        tuple: Training and testing data.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logging.info("Data split into training and testing sets.")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on the training data.
    
    Parameters:
        X_train (DataFrame): Features for training.
        y_train (Series): Target variable for training.
    
    Returns:
        model: Trained RandomForest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data and print the accuracy.
    
    Parameters:
        model: Trained model to evaluate.
        X_test (DataFrame): Features for testing.
        y_test (Series): Target variable for testing.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy * 100:.2f}%")

def main(file_path, target_column):
    """
    Main function to orchestrate data loading, preprocessing, training, and evaluation.
    
    Parameters:
        file_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target column for prediction.
    """
    data = load_data(file_path)
    if data is None:
        return
