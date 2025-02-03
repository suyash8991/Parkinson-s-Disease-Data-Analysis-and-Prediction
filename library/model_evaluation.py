import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix  
)
from config.logger_config import Logger, log_decorator
import traceback as tb

class ModelEvaluation:
    def __init__(self, model_data, logger=None):
        """
        Initialize with comprehensive model data for evaluation.
        
        Args:
            model_data (dict): Dictionary containing model, data, and training information
            logger (logging.Logger, optional): External logger instance
        """
        self.logger = logger or Logger().get_logger()
        try:
            self.model = model_data['model']
            self.X_test = model_data['X_test']
            self.y_test = model_data['y_test']
            self.model_name = model_data.get('model_name', str(self.model.__class__.__name__))
            self.best_params = model_data.get('best_params', {})
            
            self.logger.info(f"Initialized Model Evaluation for {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Model Evaluation: {str(e)}")
            self.logger.error(tb.format_exc())
            raise

    @log_decorator
    def evaluate(self, detailed=True):
        """
        Comprehensive model evaluation with multiple metrics and visualizations.
        
        Args:
            detailed (bool): Flag to generate detailed evaluation including plots
        
        Returns:
            dict: Comprehensive evaluation metrics and optionally visualization data
        """
        try:
            # Predictions
            y_pred = self.model.predict(self.X_test)
            
            # Probabilistic predictions (for binary classification)
            try:
                y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            except Exception:
                y_pred_proba = None
            
            cm = confusion_matrix(self.y_test, y_pred,labels=[1,0])

            
            # Basic metrics
            results = {
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
                'model_name': self.model_name,
                'best_hyperparameters': self.best_params,
                'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization

            }
            
            # Advanced metrics and visualizations if probabilistic predictions available
            if y_pred_proba is not None:
                results.update(self._compute_probability_metrics(y_pred_proba))
                
                if detailed:
                    results.update(self._generate_visualizations(y_pred_proba))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            self.logger.error(tb.format_exc())
            raise


    def _compute_probability_metrics(self, y_pred_proba):
        """
        Compute probability-based metrics.
        
        Args:
            y_pred_proba (np.ndarray): Predicted probabilities
        
        Returns:
            dict: Probability-based metrics
        """
        metrics = {}
        try:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(self.y_test, y_pred_proba)
        except :
            self.logger.error("Failed to compute probability metrics: ",tb.format_exc())
        
        return metrics

    def _generate_visualizations(self, y_pred_proba):
        """
        Generate ROC and Precision-Recall curves.
        
        Args:
            y_pred_proba (np.ndarray): Predicted probabilities
        
        Returns:
            dict: Visualization data or paths
        """
        visuals = {}
        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            
            plt.figure(figsize=(12, 5))
            
            # ROC Subplot
            plt.subplot(121)
            plt.plot(fpr, tpr, color='darkorange', lw=2)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title(f'ROC Curve - {self.model_name}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            
            # Precision-Recall Subplot
            plt.subplot(122)
            plt.plot(recall, precision, color='blue', lw=2)
            plt.title(f'Precision-Recall Curve - {self.model_name}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            
            plt.tight_layout()
            plt.savefig(f'results/{self.model_name}_evaluation_curves.png')
            plt.close()
            
            visuals['curve_plot_path'] = f'results/{self.model_name}_evaluation_curves.png'
        
        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {str(e)}")
        
        return visuals