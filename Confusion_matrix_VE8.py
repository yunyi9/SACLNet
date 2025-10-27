import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':

    conf_matrix = torch.tensor([[23, 2, 2, 4, 1, 0, 1, 0],
        [ 1, 9, 6, 3, 5, 2, 5, 2],
        [ 2, 1, 21, 6, 3, 2, 3, 1],
        [ 2, 4, 4, 39, 3, 1, 1, 0],
        [ 4, 2, 4, 5, 35, 2, 4, 4],
        [ 1, 1, 0, 6, 1, 22, 0, 2],
        [ 2, 4, 2, 4, 5, 3, 60, 1],
        [ 2, 0, 3, 3, 8, 0, 4, 12]])

    labels = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
    draw_confusion_matrix_from_matrix(conf_matrix, labels, 've8_confusion.jpg', dpi=1000)
