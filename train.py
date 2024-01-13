import argparse
import json

import torch
from torch import nn, optim
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from collections import OrderedDict


def build_model(arch, hidden_units, learning_rate):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError("Please select: vgg19/vgg16/alexnet")

    if arch == 'vgg19' or arch == 'vgg16':
        in_features = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features, hidden_units)), ('dropout1', nn.Dropout(0.5)), ('relu1', nn.ReLU()),
            ('l2', nn.Linear(hidden_units, 102)),
            ('output_layer', nn.LogSoftmax(dim=1))]))
        model.classifier = classifier

    elif arch == 'alexnet':
        in_features = model.classifier[1].in_features
        classifier = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(in_features, hidden_units)), ('dropout1', nn.Dropout(0.5)), ('relu1', nn.ReLU()),
            ('l2', nn.Linear(hidden_units, 102)),
            ('output_layer', nn.LogSoftmax(dim=1))]))
        model.classifier = classifier
    return model, classifier


def save_checkpt(model, hidden_units, epochs, optimizer, path):
    checkpoint = {'arch': 'vgg19',
                  'state_idx': model.state_dict(),
                  'class_to_idx': imagefolder_datasets[0].class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'classifier': classifier,
                  'hidden_units': hidden_units,
                  'epochs': epochs
                  }

    torch.save(checkpoint, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training part of COMMAND LINE APPLICATION")
    parser.add_argument("data_dir", type=str)
    parser.add_argument('--save_dir', type=str, action='store', default='checkpt.pth')
    parser.add_argument('--arch', type=str, default='vgg19', choices=['vgg19', 'vgg16', 'alexnet'])
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    data_dir = args.data_dir
    path = 'checkpt.pth'

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    training_transform = transforms.Compose([transforms.RandomResizedCrop(size=244),
                                             transforms.RandomRotation(30),
                                             transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    validation_transform = transforms.Compose([transforms.Resize(size=255),
                                               transforms.CenterCrop(size=224),
                                               transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([transforms.Resize(size=255),
                                            transforms.CenterCrop(size=224),
                                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std=[0.229, 0.224, 0.225])])

    imagefolder_datasets = [datasets.ImageFolder(train_dir, transform=training_transform),
                            datasets.ImageFolder(valid_dir, transform=validation_transform),
                            datasets.ImageFolder(test_dir, transform=testing_transform)]

    dataloader_sets = [torch.utils.data.DataLoader(imagefolder_datasets[0], batch_size=64, shuffle=True),
                       torch.utils.data.DataLoader(imagefolder_datasets[1], batch_size=64, shuffle=True),
                       torch.utils.data.DataLoader(imagefolder_datasets[2], batch_size=64, shuffle=True)]

    model, classifier = build_model(args.arch, args.hidden_units, args.learning_rate)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)

    steps = 0
    freq = 10
    for epoch in range(args.epochs):

        model.train()
        train_loss = 0
        for inputs, labels in dataloader_sets[0]:
            steps += 1

            if args.gpu:
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

            if steps % freq == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0

                for v_inputs, v_labels in dataloader_sets[1]:
                    optimizer.zero_grad()
                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                    model.to(device)

                    with torch.no_grad():
                        output = model.forward(v_inputs)
                        valid_loss = criterion(output, v_labels)
                        ps = torch.exp(output).data
                        top_prob, top_class = ps.topk(1, dim=1)
                        cal = top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(cal.type(torch.FloatTensor)).item()

                (print(f"Epoch=> {epoch + 1}/{args.epochs} ",
                       f"Training=> Loss: {train_loss / freq:.3f}  ",
                       f"Validation=> Loss: {valid_loss / len(dataloader_sets[1]):.3f} Accuracy: {accuracy / len(dataloader_sets[1]):.4f}"))

                train_loss = 0

        save_checkpt(model, optimizer, args.epochs, path, classifier)
