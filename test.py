import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from models.LeNet import LeNet
from utils import process

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_single(model, filename, plot=False):
    ret_images = process(filename)

    if not 4 <= len(ret_images) <= 5:
        return 'ERROR'

    # standardize
    for j in range(len(ret_images)):
        ret_images[j] = (ret_images[j] - np.mean(ret_images[j])) / (0.0001 + np.std(ret_images[j]))

    tensor_images = torch.from_numpy(np.array(ret_images)).float().to(DEVICE)
    tensor_images = tensor_images.unsqueeze(1)

    if plot:
        grig_images = make_grid(tensor_images, normalize=True)
        plt.imshow(grig_images.permute([1, 2, 0]))
        plt.show()

    pred = F.softmax(model(tensor_images), dim=1)
    labels = pred.max(1)[1].detach()

    return ''.join([chr(ord('a') + x.item()) for x in labels])


def test_multiple(model):
    with open('data/JNIST/label_validate.csv', 'r') as f:
        lines = f.readlines()

    correct = 0

    for i in range(len(lines)):
        target_label = ''.join(lines[i][:-1].split(',')[1:])
        filename = 'data/JNIST/validate/%05d.jpg' % i
        pred_label = test_single(model, filename)

        if target_label == pred_label:
            correct += 1
        else:
            print(filename, target_label, pred_label)

    print('Test accuracy on validate dataset is %.3f' % (correct / len(lines)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=int, help='model=0, 1, 2, 3', required=True)
    parser.add_argument('-f', '--filename', type=str, help='filename', required=False)

    args = parser.parse_args()

    if args.model == 3:
        model = LeNet().to(DEVICE)
        model.load_state_dict(torch.load('./models/lenet.pt'))
        model.eval()
    else:
        model = None

    # args.filename = 'data/JNIST/test/00029.jpg'
    if args.filename is not None:
        print(test_single(model, args.filename, True))
    else:
        test_multiple(model)
