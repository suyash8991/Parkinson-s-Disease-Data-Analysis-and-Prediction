import joblib
import os 
from constant.constant import MODEL_SAVE_PATH
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
