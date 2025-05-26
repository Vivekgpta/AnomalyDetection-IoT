# Helper utilities (e.g., plotting, formatting)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels):
    """
    Plots a confusion matrix using Seaborn heatmap.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
