import argparse
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from models.gnn import GCN, GraphSAGE
from models.gin import GINModel
from params import *
from utils import *
from torch_geometric.data import NeighborSampler
from torch.utils.data import Subset
from models.pna import *
from torch_geometric.utils import degree

def evaluateFinalModel():
    model = loadModel(final=True)
    print(f"Evaluating {FINAL_MODEL_PATH}...")
    modelAccuracy(model, test_loader, save=False)

def evaluateModel():
    model = loadModel(final=False)
    print(f"Evaluating {MODEL_PATH}...")
    modelAccuracy(model, test_loader, save=False)

def prepareLoaders(dataset):
    batch_size = BATCH_SIZE
    indices = torch.randperm(len(dataset))

    train_indices = indices[250:950]  # Training indices
    test_indices = torch.cat((indices[:200], indices[950:1000]))  # Test indices

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset

def prepareTraining(random, model_type=None, deg=None):
    if model_type == 'gcn':
        model = GCN(random)
    elif model_type == 'graphsage':
        model = GraphSAGE(random)
    elif model_type == 'gin':
        model = GINModel()
    else:
        model = PNA(random, deg)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, optimizer

def train(model, train_loader, optimizer, random, num_samples):
    model.train()

    total_loss = 0
    i = 0
    for data in train_loader:
        optimizer.zero_grad()
        if random:
            # x is size N x 1 x M for M samples
            x = torch.randn((data.x.shape[0], 1, num_samples)).to('cuda')
            edge_index = data.edge_index.to('cuda')
        else:
            # x is size N x F
            x = data.x.to('cuda')
            edge_index = data.edge_index.to('cuda')
        pred = model(x, edge_index, data.batch.to('cuda'))
        loss = loss_function(pred, data.y.float().to('cuda'))
        loss.backward()
        optimizer.step()

        total_loss += loss
    return total_loss

def trainingLoop(model, train_loader, optimizer, log=True, logExtended = False, save_model=True, random=True, num_samples=None):
    print("Training model...")
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, optimizer, random, num_samples)

        if log and epoch % 5 == 0:
                avg_acc = accuracy(train_loader, model, random, num_samples)
                print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Accuracy: {avg_acc:.2f}')

    if (save_model):
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model was saved to {MODEL_PATH}")

def modelAccuracy(model, test_loader, save=False, random=True, num_samples=10):
    print("Model accuracy for random=", random)
    avg_acc = accuracy(test_loader, model, random, num_samples=num_samples)
    print(f'Model Accuracy: {avg_acc:.2f}')

    return avg_acc

def trainAndSaveModel(random, num_samples, model_type, train_dataset):
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    if num_samples is not None:
        num_samples = int(num_samples)
    
    model, optimizer = prepareTraining(random, model_type, deg)
    model = model.to('cuda')

    trainingLoop(model, train_loader, optimizer, random=random, num_samples=num_samples)
    modelAccuracy(model, test_loader, random=random, num_samples=num_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-evalFinal', action='store_true')    
    parser.add_argument('-evalTrained', action='store_true')
    parser.add_argument('-test', action='store_true')   
    parser.add_argument('-train', action='store_true')   
    parser.add_argument('-random', action='store_true')
    parser.add_argument('-num_samples', nargs='+', type=int)
    parser.add_argument('-model_type', type=str)
    args = parser.parse_args()

    # Random denotes random sampling for node features
    # num_samples specifies M, the number of random samples
    dataset = loadDataset()
    train_loader, test_loader, train_dataset = prepareLoaders(dataset)
    loss_function = torch.nn.BCELoss() #CrossEntropYloss

    if args.evalFinal:
        evaluateFinalModel()
    elif args.evalTrained:
        evaluateModel()
    elif args.train:
        if args.num_samples is None:
            trainAndSaveModel(args.random, args.num_samples, args.model_type, train_dataset)
        else:
            for num_sample in args.num_samples:
                print("Num samples:", num_sample)
                trainAndSaveModel(args.random, num_sample, args.model_type)
    elif args.test:
        testModelAccuracy(int(args.num_samples))
    else:
        print("No arguments were specified. Please run this file with the correct flags (see documentation).")
