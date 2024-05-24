import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.utils import degree
from torch_geometric.explain import Explainer, GNNExplainer
from PIL import Image, ImageDraw, ImageFont

from params import *

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
def accuracy(loader, model, random=True, num_samples=10, acc_type=None):
    model.eval()

    if acc_type == 'labels':
        total_acc = 0
        for data in loader:
            if random:
                x = torch.randn((data.x.shape[0], 1, num_samples)).to('cuda')
                #x = data.x.view(data.x.shape[0], data.x.shape[1], num_samples)
            else:
                x = data.x.to('cuda')
            _, labels = model.evaluate(x, data.edge_index.to('cuda'), data.batch.to('cuda'))
            total_acc += int((labels == data.y.to('cuda')).sum())
        print(len(loader.dataset))
        total_acc /= len(loader.dataset)

        return total_acc
    else:
        total_error = 0
        for data in loader:
            data = data.to('cuda')
            if random:
                x = torch.randn((data.x.shape[0], 1, num_samples)).to('cuda')
            else:
                x = data.x
            out = model.evaluate(x, data.edge_index, data.batch)
            total_error += (out.squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)


def loadModel(final=True):
    from nets import GCN
    model = GCN()
    if final:
        model.load_state_dict(torch.load(FINAL_MODEL_PATH))
    else:
        model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()

    return model

def loadDataset(type=None):
    if type == 'imdb':
        dataset = TUDataset("./", "IMDB-BINARY")
        initializeNodes(dataset)
    else:
        dataset = ZINC('./', "datasets/ZINC")
        initializeNodes(dataset)

    return dataset
