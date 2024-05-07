import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.explain import Explainer, GNNExplainer
from PIL import Image, ImageDraw, ImageFont

from parameters import *

class AddDegreeAsFeature:
    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        data.x = deg.view(-1, 1)
        return data
    
def initializeNodes(dataset):
    dataset.transform = AddDegreeAsFeature()


def label(pred):
    pred[pred > P_THRESHOLD] = 1
    pred[pred <= P_THRESHOLD] = 0

    return pred

@torch.no_grad()
def accuracy(loader, model, random=True, num_samples=10):
    model.eval()

    total_acc = 0
    for data in loader:
        if random:
            x = torch.randn((data.x.shape[0], 1, num_samples))
            #x = data.x.view(data.x.shape[0], data.x.shape[1], num_samples)
        else:
            x = data.x
        _, labels = model.evaluate(x, data.edge_index, data.batch)
        total_acc += int((labels == data.y).sum())
    print(len(loader.dataset))
    total_acc /= len(loader.dataset)

    return total_acc

def loadModel(final=True):
    from nets import GCN
    model = GCN()
    if final:
        model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    return model

def loadDataset():
    dataset = TUDataset("./", "IMDB-BINARY")
    initializeNodes(dataset)

    return dataset
