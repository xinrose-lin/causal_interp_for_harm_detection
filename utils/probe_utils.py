from tqdm import tqdm
import torch as t
from torch import nn
import random
from utils.data_utils import read_from_pt_gz

## linear
class Probe(nn.Module):
    def __init__(self, activation_dim):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=True)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits

def load_probe(filepath, dim):
    
    model = Probe(dim)
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
        (t.tensor(data[i:i+batch_size]), t.tensor(labels[i:i+batch_size], device=device)) for i in range(0, len(data), batch_size)
    ]
    return batches


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

#### experiments code compiled

def get_train_test_batches(nonharmful_fp, harmful_fp): 
    ### get train acts
    nonharmful_acts = read_from_pt_gz("data_latents/train/acts/nonharmful_acts_512.pt.gz")
    harmful_acts = read_from_pt_gz("data_latents/train/acts/harmful_acts_512.pt.gz")

    last_tok_acts_data = t.cat((nonharmful_acts[:, -1, :], harmful_acts[:, -1, :]), dim=0).tolist()
    label = t.cat( (t.zeros(len(nonharmful_acts)) , t.ones(len(harmful_acts))) )

    train_batches = data_loader(last_tok_acts_data, label)

    ## get test acts
    nonharmful_acts = read_from_pt_gz("data_latents/test/acts/nonharmful_acts_512.pt.gz")
    harmful_acts = read_from_pt_gz("data_latents/test/acts/harmful_acts_512.pt.gz")

    last_tok_acts_data = t.cat((nonharmful_acts[:, -1, :], harmful_acts[:, -1, :]), dim=0).tolist()
    label = t.cat( (t.zeros(len(nonharmful_acts)) , t.ones(len(harmful_acts))) )
    test_batches = data_loader(last_tok_acts_data, label)
    
    return train_batches, test_batches


def train(linear_probe_dim, train_batches, test_batches): 
    t.manual_seed(42)
    probe = Probe(linear_probe_dim)

    epoch_train_loss = []
    total_loss = []
    epoch_test_acc = []
    epoches = 65

    for i in range(epoches): 
        probe, losses = train_probe(probe, train_batches)
        total_loss.extend(losses)
        epoch_train_loss.append(losses[-1])

        test_acc = test_probe(probe, batches=test_batches, seed=42)
        epoch_test_acc.append(test_acc)

    return probe, epoch_train_loss, epoch_test_acc


def test_scores(probe, test_batches):
    ## get test acts
    nonharmful_acts = read_from_pt_gz("data_latents/test/acts/nonharmful_acts_512.pt.gz")
    harmful_acts = read_from_pt_gz("data_latents/test/acts/harmful_acts_512.pt.gz")

    last_tok_acts_data = t.cat((nonharmful_acts[:, -1, :], harmful_acts[:, -1, :]), dim=0).tolist()
    label = t.cat( (t.zeros(len(nonharmful_acts)) , t.ones(len(harmful_acts))) )
    test_batches = data_loader(last_tok_acts_data, label)
    
    ## accuracy
    accuracy = test_probe(probe, batches=test_batches, seed=42)

    ## recall (harmful)
    harmful_acts = read_from_pt_gz("data_latents/test/acts/harmful_acts_512.pt.gz")
    last_tok_acts_data = harmful_acts[:, -1, :].tolist()
    label = t.ones(len(harmful_acts))
    test_batches_harmful = data_loader(last_tok_acts_data, label)

    recall = test_probe(probe, test_batches_harmful)

    ## recall (perturbed)
    harmful_acts = read_from_pt_gz("data_latents/test_perturbed/acts/harmful_acts_512.pt.gz")
    last_tok_acts_data = harmful_acts[:, -1, :].tolist()
    label = t.ones(len(harmful_acts))
    test_batches_harmful_perturbed = data_loader(last_tok_acts_data, label)

    recall_perturbed = test_probe(probe, test_batches_harmful_perturbed)

    ## recall score (nonharmful)
    nonharmful_acts = read_from_pt_gz("data_latents/test/acts/nonharmful_acts_512.pt.gz")
    last_tok_acts_data = nonharmful_acts[:, -1, :].tolist()

    label = t.zeros(len(nonharmful_acts))
    test_batches_nonharmful = data_loader(last_tok_acts_data, label)

    recall_nonharmful = test_probe(probe, test_batches_nonharmful)

    return accuracy, recall, recall_perturbed, recall_nonharmful