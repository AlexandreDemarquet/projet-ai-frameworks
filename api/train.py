import torch
import argparse
from statistics import mean
from torch import nn, optim
from tqdm import tqdm
from model import FilmClassifier
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def train(net, optimizer, loader, epochs=10):
#     criterion = nn.CrossEntropyLoss()
#     for epoch in range(epochs):
#         running_loss = []
#         t = tqdm(loader)
#         for x, y in t:
#             x, y = x.to(device), y.to(device)
#             outputs = net(x)
#             loss = criterion(outputs, y)
#             running_loss.append(loss.item())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             t.set_description(f'training loss: {mean(running_loss)}')


def train(net, optimizer, loader, epochs, freeze_epochs, lr, fn_lr):
    criterion = nn.CrossEntropyLoss()

    # Freeze all layers except the final fully connected layer for the first `freeze_epochs`
    for param in net.model.parameters():
        param.requires_grad = False
    # Unfreeze the final layer
    for param in net.model.fc.parameters():
        param.requires_grad = True

    # First phase: Train only the final fully connected layer (with larger learning rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.model.parameters()), lr=lr)
    
    # Train for `freeze_epochs` using the larger learning rate
    for epoch in range(freeze_epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(net.device), y.to(net.device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'Phase 1 - Epoch {epoch + 1}/{freeze_epochs}, Loss: {mean(running_loss):.4f}')
    
    # Second phase: Unfreeze all layers and fine-tune with smaller learning rate
    for param in net.model.parameters():
        param.requires_grad = True
    
    # Now use a smaller learning rate for fine-tuning
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.model.parameters()), lr=fn_lr)
    
    # Fine-tune the entire model for the remaining epochs
    for epoch in range(freeze_epochs, epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(net.device), y.to(net.device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'Phase 2 - Epoch {epoch + 1}/{epochs}, Loss: {mean(running_loss):.4f}')


def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default = 'MNIST', help='experiment name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--fine_tuning_lr', type=float, default=1e-5, help='fine-tuning learning rate')
    parser.add_argument('--freeze_epochs', type=int, default=10, help='number of epochs to freeze the model')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epoch
    batch_size =  args.batch_size
    lr =  args.lr
    fn_lr = args.fine_tuning_lr
    freeze_epochs = args.freeze_epochs

    # transforms
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    dataset = torchvision.datasets.ImageFolder(root="content/sorted_movie_posters_paligema", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    model = FilmClassifier(10)
    # setting net on device(GPU if available, else CPU)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    #train(model, optimizer, trainloader, epochs)
    train(model, optimizer, trainloader, epochs, freeze_epochs, lr, fn_lr)
    test_acc = test(model, testloader)
    print(f'Test accuracy:{test_acc}')

    #torch.save(net.state_dict(), 'weights/mnist_net.pth')
    torch.save(model.state_dict(), 'weights/filmClassifier.pth', _use_new_zipfile_serialization=False)
