import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from constant.constant import DATA
from library.loader import DataLoader
from library.eda import ParkinsonEDA
from library.data_preprocessing import DataPreprocessing

def main():
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Step 1: Data Loading and EDA
    data_loader = DataLoader()
    data = data_loader.load_csv(DATA)
    
    eda = ParkinsonEDA(data)
    eda.ydata_profiling()
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

    # Prepare data for LazyPredict
    X = processed_data.drop('status', axis=1)
    y = processed_data['status']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )

    # Initialize and fit LazyClassifier
    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True, 
        custom_metric=None,
        predictions=True
    )
    
    models_train, predictions_train = clf.fit(X_train, X_test, y_train, y_test)

    # Generate detailed report
    models_train.to_csv('results/lazy_predict_model_comparison.csv')

    # Detailed model performance report
    with open('results/lazy_predict_model_report.txt', 'w') as report_file:
        report_file.write("LazyPredict Model Comparison Report\n")
        report_file.write("=" * 40 + "\n\n")
        
        report_file.write(str(models_train))
        
        report_file.write("\n\nDetailed Model Predictions:\n")
        for model_name, model_predictions in predictions_train.items():
            report_file.write(f"\n{model_name} Predictions:\n")
            report_file.write(str(model_predictions))

    # Optional: Visualize model performances
    try:
        import matplotlib.pyplot as plt
        
        # Reset index to make plotting easier
        models_df = models_train.reset_index()
        
        plt.figure(figsize=(15, 8))
        
        # Create a bar plot with multiple metrics
        metrics = ['Accuracy', 'Balanced Accuracy', 'ROC AUC']
        x = np.arange(len(models_df))
        width = 0.25
        
        plt.bar(x - width, models_df['Accuracy'], width, label='Accuracy')
        plt.bar(x, models_df['Balanced Accuracy'], width, label='Balanced Accuracy')
        plt.bar(x + width, models_df['ROC AUC'], width, label='ROC AUC')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models_df['index'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/lazy_predict_model_performances.png')
        plt.close()
    except ImportError:
        print("Matplotlib not available for visualization")
    except Exception as e:
        print(f"Error in visualization: {e}")

    print("LazyPredict model training and evaluation complete. Check 'results' directory for detailed reports.")

if __name__ == "__main__":
    main()