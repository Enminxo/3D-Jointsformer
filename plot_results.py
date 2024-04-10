import itertools
import pickle
import json
import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(path,
                          cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    out_path = '{}/{}.png'.format(path, title)
    plt.savefig(out_path)


def get_logs(logs_path):
    logs = []
    train_logs = []
    val_logs = []
    with open(logs_path, 'r') as file:
        for log in file:
            logs.append(json.loads(log))

    for data in logs[1:]:
        if data['mode'] == 'train':
            train_logs.append(data)
        if data['mode'] == 'val':
            val_logs.append(data)

    epochs = [i['epoch'] for i in train_logs if 'epoch' in i.keys()]
    train_loss = [i['loss'] for i in train_logs if 'loss' in i.keys()]
    valid_loss = [i['loss'] for i in val_logs if 'loss' in i.keys()]
    train_acc = [i['top1_acc'] for i in train_logs if 'top1_acc' in i.keys()]
    valid_acc = [i['top1_acc'] for i in val_logs if 'top1_acc' and 'loss' in i.keys()]

    return epochs, train_acc, valid_acc, train_loss, valid_loss



def save_plots(path, title, train_acc, valid_acc, train_loss, valid_loss):
    """

    Args:
        path:
        title: model + loss or acc
    Returns:

    """

    if not os.path.exists(path):
        os.makedirs(path)
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='--',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='--',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Valid Accuracy')
    acc_path = '{}/{}_{}.png'.format(path, title, 'Accuracy')
    plt.savefig(acc_path)

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train vs Valid Loss')
    loss_path = '{}/{}_{}.png'.format(path, title, 'Loss')
    plt.savefig(loss_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='This script save the confusion matrix of the testing results')
    parser.add_argument('--logs_file', type=str, default=None,
                        help='path to train logs file:json file')
    parser.add_argument('--output', type=str, default='output/briareo', help='path to save the plots')
    parser.add_argument('--task', type=str, default='RGB')
    parser.add_argument('--plot_title', type=str, help=' title of the output plot')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    task_pth = os.path.join(args.output, args.task)
    os.makedirs(task_pth, exist_ok=True)

    """
    a confusion matrix C is such that Ci,j is equal to the number of observations known to be in group i and predicted to be in group j.
    gt, pred = get_preds(args.input)
    cm = confusion_matrix(y_true=gt, y_pred=pred)
    cm_plot_labels = ['g00', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11']
    plot_confusion_matrix(path=task_pth, cm=cm, classes=cm_plot_labels, title=args.cm_title)
    """

    # todo: save the loss and acc learning curve
    epoch, train_acc, valid_acc, train_loss, valid_loss = get_logs(logs_path=args.logs_file)
    save_plots(task_pth, args.plot_title, train_acc, valid_acc, train_loss, valid_loss)
