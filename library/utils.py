import joblib
import os 
from constant.constant import MODEL_SAVE_PATH
import numpy as np
def save_model(model, file_name):
    """
    Save model to a file using joblib.
    """
    model_path = os.path.join(MODEL_SAVE_PATH, file_name)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(file_name):
    """
    Load model from a file.
    """
    model = joblib.load(file_name)
    print(f"Model loaded from {file_name}")
    return model


def format_confusion_matrix_report(cm, labels):
    """
    Generate a detailed confusion matrix report with labels.
    
    Args:
        cm (list or np.ndarray): Confusion matrix
        labels (list): List of class labels
    
    Returns:
        str: Formatted confusion matrix report
    """
    # Convert to numpy array for easier manipulation
    cm_array = np.array(cm)
    total_samples = cm_array.sum()
    
    # Create formatted report
    report = "Confusion Matrix Report:\n"
    report += "=" * 30 + "\n\n"
    
    # Header
    header = f"{'Predicted ->':>15} | " + " | ".join(f"{label:>10}" for label in labels)
    report += header + "\n"
    report += "-" * len(header) + "\n"
    
    # Matrix with labels
    for i, true_label in enumerate(labels):
        # Create row with absolute counts
        row_counts = cm_array[i]
        count_str = f"{true_label} (Actual) | " + " | ".join(f"{count:>10}" for count in row_counts)
        
        # Calculate percentages for this row
        row_percentages = (row_counts / row_counts.sum() * 100).round(2)
        percentage_str = f"{'':>15} | " + " | ".join(f"{pct:>10.2f}%" for pct in row_percentages)
        
        report += count_str + "\n"
        report += percentage_str + "\n\n"
    
    # Performance metrics
    correct_predictions = np.trace(cm_array)
    misclassifications = total_samples - correct_predictions
    
    report += "Performance Summary:\n"
    report += f"Total Samples:            {total_samples}\n"
    report += f"Correct Predictions:      {correct_predictions} ({correct_predictions/total_samples*100:.2f}%)\n"
    report += f"Misclassifications:       {misclassifications} ({misclassifications/total_samples*100:.2f}%)\n"
    
    return report
