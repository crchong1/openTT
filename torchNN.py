import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import csv
import random
import os
import librosa
from sklearn.model_selection import cross_val_score
import skorch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
URBAN_SOUND_CSV_PATH = './UrbanSound8K.csv'
URBAN_SOUND_FOLDER = './sampleUrbanSound8k'
PING_PONG_SOUND_FOLDER = './samplePingPongSounds'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = .01
log_interval = 200
num_classes = 12 # 2 classes of data- either ping pong or not ping pong
train_test_split_ratio = .7
PATH = './opentt.pth'
def main():
    labelMap = getLabels()
    soundData, labelData = formatWav(labelMap)
    loaderTrain, loaderTest = loader(soundData, labelData)
    model, optimizer, scheduler = setModel()
    runModel(model, scheduler, loaderTrain, loaderTest, optimizer)

def getLabels():
    f = open(URBAN_SOUND_CSV_PATH)
    csv_file = csv.reader(f)
    labels = {}
    for row in csv_file:
        filename = row[0]
        classLabel = row[6]
        labels[filename] = classLabel
    return labels

def formatWav(labelMap):
    filenames = []
    for filename in os.listdir(URBAN_SOUND_FOLDER):
        if filename.endswith(".wav"):
            filenames.append(filename)
    for filename in os.listdir(PING_PONG_SOUND_FOLDER):
        if filename.endswith(".wav"):
            filenames.append(filename)
    random.shuffle(filenames)
    soundData = []
    labelData = []
    for filename in filenames:
        print(filename)
        path = ''
        if("chunk" in filename):
            path = str(PING_PONG_SOUND_FOLDER + os.sep + filename)
        else:
            path = str(URBAN_SOUND_FOLDER + os.sep + filename)
        data, rate = librosa.load(path, mono=True, duration=5.0)
        data = 2.*(data - np.min(data))/np.ptp(data)-1 # normalize between -1 and 1
        if(data.shape[0] < 110250):
            data = np.pad(data, (0,110250-data.shape[0]), mode="symmetric")
        label = ""
        if(filename not in labelMap):
            label = 11
        else:
            label = int(labelMap[filename])
        soundData.append(data)
        labelData.append(label)
    return np.array(soundData), np.array(labelData)

class Dataset(Dataset):
    def __init__(self, X, y):
        self.len = len(X)
        if torch.cuda.is_available():
            self.x_data = torch.from_numpy(X).float().cuda()
            self.y_data = torch.from_numpy(y).float().cuda()
        else:
            self.x_data = torch.from_numpy(X).float()
            self.y_data = torch.from_numpy(y).float()
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.x_data[idx], self.y_data[idx])


def loader(soundData, labelData):
    splitIndex = int(soundData.shape[0]*train_test_split_ratio)
    print(splitIndex)
    datasetTrain = Dataset(soundData[0:splitIndex], labelData[0:splitIndex]) 
    loaderTrain = torch.utils.data.DataLoader(
        datasetTrain,
        batch_size=batch_size,
        shuffle=False,
    )
    datasetTest = Dataset(soundData[splitIndex:], labelData[splitIndex:])
    loaderTest = torch.utils.data.DataLoader(
        datasetTest,
        batch_size=batch_size,
        shuffle=False,
    )
    return loaderTrain, loaderTest

def setModel():
    model = CNN()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
    return model, optimizer, scheduler

# 110250
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 110250 -> 22050
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=5, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU())
        # 22050 -> 1470
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3))
        # 1470 -> 490
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU())
        # 490 -> 49
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=2, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(5,stride=5))
        self.fc = nn.Linear(256*49, num_classes) # reduction to however many features - let us make 64 features * 
        self.activation = nn.ReLU()

    def forward(self, x):
        # x : 10 x 110250 x 1
        x = x.reshape(x.shape[0], 1 ,-1)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

def train(model, epoch, train_loader, optimizer):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device=device, dtype=torch.int64)
        data = data.requires_grad_()
        output = model(data)
        # print(data.detach().cpu().numpy(), 'data')
        # print(label.detach().cpu().numpy(), 'label')
        # print(output.detach().cpu().numpy(), 'output')
        loss = F.cross_entropy(output, label) #the loss functions expects a batchSizex12 input
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

def test(model, test_loader):
    model.eval()
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        predictions = predicted.data.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        correct += (predictions == target).sum()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def runModel(model, scheduler, loaderTrain, loaderTest, optimizer):
    for epoch in range(1, 10):
        scheduler.step()
        train(model, epoch, loaderTrain, optimizer)
    test(model, loaderTest)
    saveModel(model)

def saveModel(model):
    torch.save(model, PATH)

def predict(sound_arr):
    model = torch.load(PATH, map_location='cpu')
    model.eval()
    tensor = torch.from_numpy(sound_arr)
    tensor = tensor.to(device)
    output = model(tensor)
    _, predicted = torch.max(output, 1)
    predictions = predicted.data.detach().cpu().numpy()
    return predictions


if __name__ == "__main__":
    main()