import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import torch

def format_value(value):
    formatted_value = '{:.2f}'.format(value)
    if len(formatted_value.split('.')[1]) == 1:
        formatted_value += '0'
    return formatted_value

def draw_confusion_matrix_from_matrix(cm, label_name, pdf_save_path=None, dpi=1000, normalize=True):
    if not isinstance(cm, np.ndarray):
        cm = cm.cpu().numpy() if hasattr(cm, 'cpu') else np.array(cm)

    if normalize:
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    plt.rc('font', family='Times New Roman')
    plt.rcParams.update({'font.size': 12})
    plt.imshow(cm, cmap='Blues')

    plt.yticks(range(len(label_name)), label_name)
    plt.xticks(range(len(label_name)), label_name, rotation=45)
    plt.tight_layout()
    plt.colorbar()

    for i in range(len(label_name)):
        for j in range(len(label_name)):
            color = (1, 1, 1) if i == j else (0, 0, 0)
            value = format_value(cm[i, j])
            plt.text(j, i, value, va='center', ha='center', color=color)

    if pdf_save_path:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


conf_matrix = torch.tensor([
    [75, 5, 8, 5, 8, 14],
    [9, 82, 12, 5, 8, 5],
    [14, 10, 90, 9, 10, 4],
    [8, 7, 13, 100, 7, 19],
    [4, 2, 10, 16, 67, 8],
    [17, 17, 9, 19, 5, 117]
])

labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']

draw_confusion_matrix_from_matrix(
    cm=conf_matrix,
    label_name=labels,
    pdf_save_path='confusion_matrix_ekman6.jpg',
    dpi=300,
    normalize=True  
)