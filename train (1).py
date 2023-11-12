import numpy as np
import torch
from torch import nn, optim
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import futility
import fmodel
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.ort futility
import fmodel
import argparse

def train_model(data_dir, save_dir="./checkpoint.pth", arch="vgg16", 
                learning_rate=0.001, hidden_units=512, epochs=3, dropout=0.2, gpu=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    
    trainloader, validloader, testloader, train_data = futility.load_data(data_dir)
    
    # loop over each architecture
    if isinstance(arch, list):
        for a in arch:
            model, criterion = fmodel.setup_network(a, dropout, hidden_units, learning_rate, device)
            optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
            train_one_model(model, criterion, optimizer, trainloader, validloader, epochs, device)
    else:
        model, criterion = fmodel.setup_network(arch, dropout, hidden_units, learning_rate, device)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        train_one_model(model, criterion, optimizer, trainloader, validloader, epochs, device)
    
    # Save the checkpoint for the last trained model
    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure': arch,
                'hidden_units': hidden_units,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'no_of_epochs': epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, save_dir)
    print("Checkpoint has been saved.")
    

def train_one_model(model, criterion, optimizer, trainloader, validloader, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 5
    
    print(f"--Training ongoing for {model.__class__.__name__}--")
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
          
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for train.py')
    parser.add_argument('data_dir', action="store", default="./flowers/")
    parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
    parser.add_argument('--arch', action="store", nargs='+', default=["vgg16"])
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
    parser.add_argument('--epochs', action="store", default=3, type=int)
    parser.add_argument('--dropout', action="store", type=float, default=0.2)
    parser.add_argument('--gpu', action="store_true", default=True)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, 
                args.hidden_units, args.epochs, args.dropout, args.gpu)

    
    #used chatgpt to format and guide through with errors ---------------------------------------------