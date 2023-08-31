#!/usr/bin/python3
#===========================================
# MNIST Training
#-------------------------------------------
# Rev.0.1 2019.09.16 Munetomo Maruyama
#-------------------------------------------
# Copyrignt (C) 2019 Munetomo Maruyama
#===========================================
# based on https://github.com/pytorch/examples/tree/master/mnist

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pdb

#===========================
# Utility
#===========================
class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

#==========================
# Neural Net
#==========================
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #  1x28x28 --> 20x24x24
        x = F.max_pool2d(x, 2, 2) # 20x24x24 --> 20x12x12 
        x = F.relu(self.conv2(x)) # 20x12x12 --> 50x 8x 8
        x = F.max_pool2d(x, 2, 2) # 50x 8x 8 --> 50x 4x 4
        x = x.view(-1, 4*4*50)    # 50x 4x 4 --> 1 x (4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
#==========================
# Training
#==========================
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end="")
    print("")

#==========================
# Test Trained CNN
#==========================
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#==========================
# Main Routune
#==========================
def main():
    # Common Constants
    args = dotdict({
        'batch_size'     : 64,    # Input batch size for training
        'test_batch_size': 1000,  # Input batch size for testing
        'epochs'         : 10,    # Number of epochs to train
        'lr'             : 0.01,  # Learning Rate
        'momentum'       : 0.5,   # SGD Momentum
        'no_cuda'        : False, # Disable CUDA Training
        'seed'           : 1,     # Random Seed
        'log_interval'   : 10,    # How many batches to wait before logging training status
        'save_model'     : True,  # For Saving the current Model
    })

    # Preparation for CUDA
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    torch.manual_seed(args.seed)

    # Load MNIST Database
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor() #,
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()#,
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
        
    # Prepare CNN Model
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Train and Test
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    # Save Model Data
    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
#=======================
# Main Entry
#=======================
if __name__ == '__main__':
    main()
    
#===========================================
# End of Program
#===========================================
