from tqdm import tqdm
import torch as t
from torch import nn
import random

## linear
class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits

def load_probe(filepath):
    """ initialis e"""
    model = Probe(512)
    # Load the state dictionary into the model
    model.load_state_dict(t.load(filepath))
    # Set the model to evaluation mode (optional, for inference)
    model.eval()
    return model 

def save_probe(probe, filepath):
    """ model.pth file """
    # Save the model's state dictionary
    t.save(probe.state_dict(), filepath)
    print('saved to ', filepath)

## returns batches of activations
def data_loader(data, labels, batch_size=16, seed = 42, device="cpu"):
    """ 
    takes in data, and labels as list
    """
    idxs = list(range(len(data)))
    # creates a shuffled list 
    random.Random(seed).shuffle(idxs)
    # get data in this shuffled order
    data, labels = [data[i] for i in idxs], [labels[i] for i in idxs]
    # return the batches
    batches = [
        (data[i:i+batch_size], t.tensor(labels[i:i+batch_size], device=device)) for i in range(0, len(data), batch_size)
    ]
    return batches


# def train_probe_(batches, lr=1e-2, epochs=1, dim=512, seed=42, probe="linear"):
#     t.manual_seed(seed)
#     if probe == "linear":
#         probe = Probe(dim)
#     else: 
#         print('define probe')

#     optimizer = t.optim.AdamW(probe.parameters(), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()

#     losses = []
#     for epoch in range(epochs):
#         for batch in batches:
#             acts = batch[0]
#             labels = batch[1] 
#             logits = probe(acts)
#             loss = criterion(logits, labels.float())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())

#     return probe, losses


def train_probe(probe, batches, lr=1e-2):
    
    optimizer = t.optim.AdamW(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for batch in batches:
        
        acts = batch[0]
        labels = batch[1] 
        logits = probe(acts)
        loss = criterion(logits, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

    return probe, losses


def test_probe(probe, batches, seed=42):
    with t.no_grad():
        corrects = []

        for batch in batches:
            acts = batch[0]
            labels = batch[1]
            # acts = get_acts(text)
            logits = probe(acts)
            preds = (logits > 0.0).long()
            # print(logits)
            # print(preds)
            corrects.append((preds == labels).float())
        return t.cat(corrects).mean().item()
