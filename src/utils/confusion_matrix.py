import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import cv2 as cv


def get_confusion_matrix_plot(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.close()
    return figure


def confusion_matrix_plot_as_array(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    figure = get_confusion_matrix_plot(cm, class_names)
    figure.canvas.draw()
    cm_plot_array = cv.cvtColor(np.array(figure.canvas.renderer.buffer_rgba()), cv.COLOR_RGBA2RGB)
    return cm_plot_array

if __name__ == "__main__":
    y_true = [0,0,1,1,0,1,1,0]
    y_pred = [0,1,0,1,0,1,0,0]
    cm_rgb = confusion_matrix_plot_as_array(y_true, y_pred, ["Male", "Female"])

    plt.imshow(cm_rgb)
    plt.show()