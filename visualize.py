# visualize.py
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.barh([feature_names[i] for i in indices], importances[indices])
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
