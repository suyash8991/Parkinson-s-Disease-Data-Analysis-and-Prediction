import pandas as pd
import os
from constant.constant import DATA
from library.loader import DataLoader
from library.eda import ParkinsonEDA
from library.data_preprocessing import DataPreprocessing
from library.model_training import ModelTraining
from library.model_evaluation import ModelEvaluation

def main():
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Step 1: Data Loading and EDA
    data_loader = DataLoader()
    data = data_loader.load_csv(DATA)
    
    eda = ParkinsonEDA(data)
    eda.summarize_data()
    eda.check_missing_values()
    
    # Data Preprocessing
    dp = DataPreprocessing(data)
    dp.drop_columns(['name'])
    dp.standardize_features()
    dp.handle_outliers()

    # Get final processed data
    processed_data = dp.get_data()

    # Save processed data
    processed_data.to_csv('data/processed_parkinsons.csv', index=False)

    print("Training the models...")
    model_trainer = ModelTraining(processed_data)
    trained_models = model_trainer.train_models()

    # Generate a comprehensive model comparison report
    model_comparison = {}
    
    with open('results/model_comparison_report.txt', 'w') as report_file:
        report_file.write("Model Comparison Report\n")
        report_file.write("=" * 30 + "\n\n")
        
        for model_name, model_data in trained_models.items():
            report_file.write(f"Model: {model_name.upper()}\n")
            
            # Extract evaluation results
            eval_results = model_data['evaluation_results']
            
            # Summarize key metrics
            report_file.write("Classification Metrics:\n")
            report_file.write(f"Precision: {eval_results['classification_report']['weighted avg']['precision']:.4f}\n")
            report_file.write(f"Recall: {eval_results['classification_report']['weighted avg']['recall']:.4f}\n")
            report_file.write(f"F1-Score: {eval_results['classification_report']['weighted avg']['f1-score']:.4f}\n")
            
            # Add ROC AUC if available
            if 'roc_auc' in eval_results:
                report_file.write(f"ROC AUC: {eval_results['roc_auc']:.4f}\n")
            
            # Best hyperparameters
            report_file.write("\nBest Hyperparameters:\n")
            for param, value in model_data['best_params'].items():
                report_file.write(f"  {param}: {value}\n")
            
            report_file.write("\n" + "=" * 30 + "\n\n")
    
    print("Model training and evaluation complete. Check 'results' directory for detailed reports.")

if __name__ == "__main__":
    main()