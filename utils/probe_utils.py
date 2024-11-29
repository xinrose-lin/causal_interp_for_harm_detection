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
    """ initialis """
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

## dataset: 
### get train acts
# nonharmful_acts = read_from_pt_gz("sparse_acts/train/nonharmful_acts_512.pt.gz")
# harmful_acts = read_from_pt_gz("sparse_acts/train/harmful_acts_512.pt.gz")

# def dataset(): 
#     data_last_tok_acts = t.cat((nonharmful_acts[:, -1, :], harmful_acts[:, -1, :]), dim=0).tolist()
#     labels = train_nonharmful[1] + train_harmful[1]
#     return data, labels
# train_batches = data_loader(last_tok_acts_data, label)


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
        (t.tensor(data[i:i+batch_size]), t.tensor(labels[i:i+batch_size], device=device)) for i in range(0, len(data), batch_size)
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

            logits = probe(acts)
            preds = (logits > 0.0).long()

            corrects.append((preds == labels).float())
        return t.cat(corrects).mean().item()



def dataset(nonharmful_fp, harmful_fp): 
    ### get train acts
    nonharmful_acts = read_from_pt_gz("data_latents/train/acts/nonharmful_acts_512.pt.gz")
    harmful_acts = read_from_pt_gz("data_latents/train/acts/harmful_acts_512.pt.gz")

    last_tok_acts_data = t.cat((nonharmful_acts[:, -1, :], harmful_acts[:, -1, :]), dim=0).tolist()
    label = t.cat( (t.zeros(len(nonharmful_acts)) , t.ones(len(harmful_acts))) )

    train_batches = data_loader(last_tok_acts_data, label)
    len(train_batches)

    ## get test acts
    nonharmful_acts = read_from_pt_gz("data_latents/test/acts/nonharmful_acts_512.pt.gz")
    harmful_acts = read_from_pt_gz("data_latents/test/acts/harmful_acts_512.pt.gz")

    last_tok_acts_data = t.cat((nonharmful_acts[:, -1, :], harmful_acts[:, -1, :]), dim=0).tolist()
    label = t.cat( (t.zeros(len(nonharmful_acts)) , t.ones(len(harmful_acts))) )
    test_batches = data_loader(last_tok_acts_data, label)
    len(test_batches)


def train(): 
    t.manual_seed(42)
    probe = Probe(512)

    epoch_train_loss = []
    total_loss = []
    epoch_test_acc = []
    epoches = 65

    # train_batches = batches[:-2]

    for i in range(epoches): 
        probe, losses = train_probe(probe, train_batches)
        total_loss.extend(losses)
        epoch_train_loss.append(losses[-1])

        test_acc = test_probe(probe, batches=test_batches, seed=42)
        epoch_test_acc.append(test_acc)

def plot():
    plt.plot(epoch_train_loss, label="Train loss")
    plt.plot(epoch_test_acc, label="Test Accuracy")
    plt.legend()
    plt.title(f"Acts probe (dim=512) training plot (Final test accuracy = {epoch_test_acc[-1]:.4f}) ")

def test_scores()
    ## accuracy
    test_probe(probe, batches=test_batches, seed=42)

    ## recall (harmful)
    harmful_acts = read_from_pt_gz("data_latents/test/acts/harmful_acts_512.pt.gz")
    last_tok_acts_data = harmful_acts[:, -1, :].tolist()
    label = t.ones(len(harmful_acts))
    test_batches_harmful = data_loader(last_tok_acts_data, label)

    recall = test_probe(probe, test_batches_harmful)

    ## recall (perturbed)
# recall (perturbed)
    harmful_acts = read_from_pt_gz("data_latents/test_perturbed/acts/harmful_acts_512.pt.gz")
    last_tok_acts_data = harmful_acts[:, -1, :].tolist()
    label = t.ones(len(harmful_acts))
    test_batches_harmful_perturbed = data_loader(last_tok_acts_data, label)

    test_probe(probe, test_batches_harmful_perturbed)
    ## recall score (nonharmful)
    ## better at nonharmful than harmful?

    nonharmful_acts = read_from_pt_gz("data_latents/test/acts/nonharmful_acts_512.pt.gz")
    last_tok_acts_data = nonharmful_acts[:, -1, :].tolist()

    label = t.zeros(len(nonharmful_acts))
    test_batches_nonharmful = data_loader(last_tok_acts_data, label)

    test_probe(probe, test_batches_nonharmful)

    t.save(probe.state_dict(), filepath)
    print('saved to ', filepath)
