import pandas as pd
from constant.constant import DATA
from library.loader import DataLoader
from library.eda import ParkinsonEDA
from library.data_preprocessing import DataPreprocessing
from library.model_training import ModelTraining
from library.model_evaluation import ModelEvaluation

def main():
    # Step 1: Data Loading and EDA
    data_loader = DataLoader()
    data = data_loader.load_csv(DATA)
    eda = ParkinsonEDA(data)
    eda.summarize_data()
    eda.check_missing_values()
    # eda.class_distribution()
    # eda.correlation_matrix()
    data['status'].value_counts()
    # Data Preprocessing
    dp = DataPreprocessing(data)
    # Drop irrelevant columns
    dp.drop_columns(['name'])
    

    # Standardize features
    dp.standardize_features()

    # Handle outliers
    dp.handle_outliers()


    # Get final processed data
    processed_data = dp.get_data()

    # Save processed data
    processed_data.to_csv('data/processed_parkinsons.csv', index=False)

    print("Training the models...")
    model_trainer = ModelTraining(processed_data)
    trained_models = model_trainer.train_models()

    print("Evaluating the models...")
    for model_name, model_data in trained_models.items():
        print(f"Evaluating {model_name} model...")
        evaluator = ModelEvaluation(model_data["model"], model_data["X_test"], model_data["y_test"])
        evaluation_results = evaluator.evaluate()
        print(f"Evaluation results for {model_name}: {evaluation_results}\n")

if __name__ == "__main__":
    main()
