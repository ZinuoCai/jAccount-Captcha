import cv2 as cv
import numpy as np
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def process(filename, output_size=32):
    ret_images = []
    rectangals = []

    gray_img = cv.imread(filename, 0)
    ret, binary = cv.threshold(gray_img, 127, 255, cv.THRESH_BINARY_INV)

    dilate_kernel = np.ones([2, 2], np.uint8)
    dilate = cv.dilate(binary, dilate_kernel, iterations=1)

    contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        rectangals.append([x, y, w, h])

    rectangals.sort(key=lambda x: x[0])
    # merge
    i = 0
    while i < len(rectangals) - 1:
        if rectangals[i + 1][0] - rectangals[i][0] < 10:
            x1, y1, w1, h1 = rectangals[i]
            x2, y2, w2, h2 = rectangals[i + 1]
            rectangals[i] = [min(x1, x2), min(y1, y2), max(w1, w2 + x2 - x1), max(h1, h2)]
            rectangals.remove(rectangals[i + 1])
        i += 1

    # split
    i = 0
    while i < len(rectangals):
        if rectangals[i][2] >= 20:
            x, y, w, h = rectangals[i]
            rectangals[i] = [x, y, w // 2, h]
            rectangals.insert(i + 1, [x + w // 2, y, w // 2, h])
        i += 1

    for rectangal in rectangals:
        x, y, w, h = rectangal
        ret_image = dilate[y:y + h, x:x + w]

        horizontal_padding = (output_size - w) // 2
        vertical_padding = (output_size - h) // 2
        if horizontal_padding >= 0 and vertical_padding >= 0:
            ret_image = cv.copyMakeBorder(ret_image, vertical_padding, vertical_padding,
                                          horizontal_padding, horizontal_padding,
                                          cv.BORDER_CONSTANT, value=0)
        ret_image = cv.resize(ret_image, (output_size, output_size))
        ret_images.append(ret_image)
    return ret_images


def get_data(root='./data/MyData/gray_split_images/', batch_size=32, workers=0):
    image_data = torchvision.datasets.ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])
        ])
    )

    lengths = [int(len(image_data) * 0.8), len(image_data) - int(len(image_data) * 0.8)]
    train_data, test_data = random_split(image_data, lengths)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers)

    return train_loader, test_loader


def get_data_jnist(root='./data/JNIST/test_split_images/', batch_size=32, workers=0):
    print('Prepare JNIST...')
    image_data = torchvision.datasets.ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])
        ])
    )

    lengths = [int(len(image_data) * 0.8), len(image_data) - int(len(image_data) * 0.8)]
    train_data, test_data = random_split(image_data, lengths)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers)

    print('Finish preparing JNIST, %d train images and %d test images.' % (lengths[0], lengths[1]))
    return train_loader, test_loader


def train(model, criterion, optimizer, train_loader, device, print_iteration):
    model.train()

    total_correct = 0
    total_train = 0
    avg_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        model.zero_grad()
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        avg_loss += loss.detach().item()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum().item()
        total_train += images.shape[0]
        loss.backward()
        optimizer.step()

        if i % print_iteration == 0:
            print('Iteration: %d, Loss: %.3f, Accuracy: %.3f'
                  % (i, avg_loss / total_train, total_correct / total_train))

    avg_loss /= total_train
    accuracy = total_correct / total_train

    print('Train Result: Loss: %.3f, Accuracy: %.3f' % (avg_loss, accuracy))
    return accuracy


def validate(model, criterion, test_loader, device):
    model.eval()

    total_correct = 0
    total_test = 0.0
    avg_loss = 0.0

    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        avg_loss += criterion(output, labels).detach().item()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum().item()
        total_test += images.shape[0]

    avg_loss /= total_test
    accuracy = total_correct / total_test

    print('Test Result: Loss: %.3f, Accuracy: %.3f' % (avg_loss, accuracy))
    return accuracy
