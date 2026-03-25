# Getting the libraries
from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import encode_features, split_data, create_features
from src.train_model import train_model
from src.evaluate_model import evaluate

def main():
    """
    Complete ML Pipeline execution

    WHY:
    - Ensures responsibility
    - Acts as single entry point for entire workflow
    """
    print('Running full ML pipeline...')

    # Load raw Dataset
    df = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Clean data
    df = clean_data(df)

    # Feature engineering

    ## Create features
    df = create_features(df)

    ## Encoding features
    df = encode_features(df)

    # Saving processed dataset
    df.to_csv("data/processed/churn_cleaned.csv", index=False)

    # Splitting data
    X, y = split_data(df)

    # Training the dataset
    model, X_test, y_test = train_model(X, y)

    # Evaluating the model
    evaluate(model, X_test, y_test)

if __name__ == '__main__':
    main()
