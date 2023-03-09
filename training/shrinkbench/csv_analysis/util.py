import numpy as np
import matplotlib.pyplot as plt


def calculate_rank(probab, pred, true):
    sorted_probab = sorted(probab)[::-1]
    true_probab = probab[true]
    pred_probab = probab[pred]
    true_rank = sorted_probab.index(true_probab)
    pred_rank = sorted_probab.index(pred_probab)
    if true_probab == pred_probab and true != pred:  # If the percentages are equal but the algo guessed wrong
        true_rank += 1
    return true_rank + 1


def calculate_class_correct(true_label, pred_label):
    """
    Returns: 1 if model correctly predicted image, 0 if incorrect
    """
    if true_label == pred_label:
        return 1
    return 0


def view_classify(img, ps):
    """ Function for viewing an image and it's predicted classes.
    """
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
