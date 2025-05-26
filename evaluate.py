# Evaluation functions like accuracy, confusion matrix, etc.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(y_true, y_pred):
    """
    Prints accuracy, confusion matrix, and classification report.
    """
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
