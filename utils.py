import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def get_data(root='./data/gray_split_images/', batch_size=32, workers=0):
    image_data = torchvision.datasets.ImageFolder(
        root=root,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])
        ])
    )

    lengths = [int(len(image_data) * 0.8), int(len(image_data) * 0.2)]
    train_data, test_data = random_split(image_data, lengths)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=workers)

    return train_loader, test_loader


def train(model, criterion, optimizer, train_loader, device):
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
