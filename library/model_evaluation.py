from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from config.logger_config import Logger
from config.logger_config import log_decorator
import traceback as tb
class ModelEvaluation:
    def __init__(self, model, X_test, y_test):
        """
        Initialize with trained model, test data, and true labels.
        """
        self.logger = Logger().get_logger()
        try:
            self.model = model
            self.X_test = X_test
            self.y_test = y_test
            self.logger.info("Able to initialize Model Evaluation")
        except e :
            self.logger.error("Data was not able to initialize Model Evaluation ",tb.format_exc())

    @log_decorator
    def evaluate(self):
        """
        Evaluate the model using different metrics.
        """
        # Predictions and probabilities
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]  # For ROC AUC
        
        # Calculate ROC AUC score
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Print classification report
        print(f"Classification Report for {self.model}:\n")
        print(classification_report(self.y_test, y_pred))

        # Plot ROC curve
        # self.plot_roc_curve(self.y_test, y_pred_proba, roc_auc)
        
        return {
            "roc_auc": roc_auc,
            "classification_report": classification_report(self.y_test, y_pred)
        }

    @log_decorator
    def plot_roc_curve(self, y_test, y_pred_proba, roc_auc):
        """
        Plot the ROC curve.
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{self.model} ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve for {self.model}')
        plt.legend(loc='lower right')
        plt.show()
