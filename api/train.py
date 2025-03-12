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

def train(net, optimizer, loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')

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
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epoch
    batch_size =  args.batch_size
    lr =  args.lr

    # transforms
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    #dataset = ImageFolder(root="content/sorted_movie_posters_paligema", transform=transform)
    dataset = torchvision.datasets.ImageFolder(root="content/sorted_movie_posters_paligema", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # DataLoader pour le train et le test
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    # trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # testloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = FilmClassifier(10)
    # setting net on device(GPU if available, else CPU)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train(model, optimizer, trainloader, epochs)
    test_acc = test(model, testloader)
    print(f'Test accuracy:{test_acc}')

    #torch.save(net.state_dict(), 'weights/mnist_net.pth')
    torch.save(model.state_dict(), 'weights/filmClassifier.pth', _use_new_zipfile_serialization=False)
