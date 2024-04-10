import os
import torch
from torch import nn
import itertools
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

plt.style.use('ggplot')


def train(device, model, dataloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    total = 0  # counter
    counter = 0
    for data, target in tqdm(dataloader):
        counter += 1
        data, target = data.to(device, dtype=torch.double), target.to(device)
        # forward pass
        output = model(data)
        # calculate the loss
        # output = nn.Softmax(dim=-1)(output)
        loss = criterion(output, target)

        train_running_loss += loss.item()
        total += target.size(0)
        _, predicted = output.max(1)
        train_running_correct += predicted.eq(target).sum().item()

        optimizer.zero_grad()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()

    # loss and accuracy for the complete epoch
    train_loss = train_running_loss / len(dataloader.dataset)
    train_accuracy = 100. * (train_running_correct / total)
    # print('train_loss', train_running_loss / counter)
    # print('train acc', 100. * (train_running_correct / len(dataloader.dataset)))
    # train_loss = train_running_loss / counter
    # train_accuracy = 100. * (train_running_correct / len(dataloader.dataset))
    return train_loss, train_accuracy


def validate(device, model, dataloader, criterion):
    model.eval()
    print('Validation')
    val_running_loss = 0.0
    val_running_correct = 0
    total = 0  # counter
    counter = 0
    for data, target in tqdm(dataloader):
        print(counter)
        with torch.no_grad():
            counter += 1
            data, target = data.to(device, dtype=torch.double), target.to(device)
            # forward pass
            output = model(data)
            # output = nn.Softmax(dim=-1)(output)
            # calculate the loss
            loss = criterion(output, target)
            val_running_loss += loss.item()

            total += target.size(0)
            _, predicted = output.max(1)
            val_running_correct += predicted.eq(target).sum().item()

    # print('dataloader vs dataloader.dataset:', len(dataloader), len(dataloader.dataset))
    # loss and accuracy for the complete epoch
    val_loss = val_running_loss / len(dataloader.dataset)
    val_accuracy = 100. * val_running_correct / total
    # print('val loss', val_running_loss / counter)
    # print('val acc', 100. * (val_running_correct / len(dataloader.dataset)))
    # val_loss = val_running_loss / counter
    # val_accuracy = 100. * (val_running_correct / len(dataloader.dataset))
    return val_loss, val_accuracy


def evaluate_model(device, model, dataloader):
    model.eval()
    print('Testing')
    valid_running_correct = 0
    counter = 0
    for data, target in tqdm(dataloader):
        with torch.no_grad():
            counter += 1
            data, target = data.to(device, dtype=torch.double), target.to(device)
            # forward pass
            outputs = model(data)
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == target).sum().item()

    # loss and accuracy for after each epoch
    final_acc = 100. * (valid_running_correct / len(dataloader.dataset))
    return final_acc


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, 'outputs/best_model.pth')


def save_model(path, epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, path + '/final_model.pth')  # 'outputs/final_model.pth'


def save_plots(path, train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
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
    plt.savefig(path + '/accuracy.png')

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
    plt.savefig(path + '/loss.png')
