import torch
import torch.nn as nn

from torch.optim import SGD
from models.ResNet import resnet18, resnet34, resnet50
from models.LeNet import LeNet
from utils import train, validate, get_data

import argparse
import matplotlib.pyplot as plt

EPOCH = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_data()
models = [resnet18, resnet34, resnet50, LeNet]
models_name = ['resnet18', 'resnet34', 'resnet50', 'lenet']


def plot(accuracies, path):
    plt.plot(range(EPOCH), accuracies[0], 'r', label='train accuracy', )
    plt.plot(range(EPOCH), accuracies[1], 'g', label='validate accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('train v.s. validate accuracy')
    plt.savefig(path)


def train_model(i):
    ###################### training ######################
    print('Begin training...')
    print('model = %s' % models_name[i])

    best_accuracy = 0.0
    accuracies = [[], []]

    model = models[i]().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    for epoch in range(EPOCH):
        print('Epoch: %d' % epoch)
        train_accuracy = train(model, criterion, optimizer, train_loader, DEVICE)
        validate_accuracy = validate(model, criterion, test_loader, DEVICE)

        accuracies[0].append(train_accuracy)
        accuracies[1].append(validate_accuracy)

        if validate_accuracy > best_accuracy:
            best_accuracy = validate_accuracy
            torch.save(model.state_dict(), 'models/%s.pt' % models_name[i])

    plot(accuracies, 'results/result_%s.png' % models_name[i])
    print('Finish training with best accuracy %.3f.' % best_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=int, help='model=0, 1, 2, 3')
    args = parser.parse_args()
    train_model(args.model)
