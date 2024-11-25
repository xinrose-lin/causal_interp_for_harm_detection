
### load dataset
import torch as t
from datasets import load_dataset
import random
import json
import numpy as np
import gzip

### TODO: (test) for sample testing 
def load_json_file(filepath):
    with open(filepath, "r") as file:
        data = json.load(file)
    
    return data

def save_to_json_file(filepath, data):
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4) 

### saving to compressed 
def save_to_pt_gz(filepath, data): 
    # write binary: massive savings on storage
    # if dict: wt (write to text)
    # if tensor: wb (write to binary)
    with gzip.open(f'{filepath}', "wb") as f:
        t.save(data, f)
    print('saved to ', filepath)

def read_from_pt_gz(filepath): 
    with gzip.open(f'{filepath}', "rb") as f:
        return t.load(f)



def load_concept_ds():
    dataset = load_dataset("justinphan3110/harmful_harmless_instructions")
    return dataset

## returns batches of activations
def data_loader(data, labels, batch_size=16, seed = 42, device="cpu"):
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


# #### for training concept probe
# def data_loader(train=True, batch_size=16, seed=42, DEVICE="cpu"):
#     """
#         load data in batches randomly for training concept probe
#     """
#     dataset = load_concept_ds()

#     if train:
#         data_split = dataset['train']
#     else:
#         data_split = dataset['test']
    
#     ### Data processing: 
#     ## flatten dataset [list of sentences] + [list of labels]
#     data = [item for sublist in data_split['sentence'] for item in sublist]
#     labels = [item for sublist in data_split['label'] for item in sublist]
    
#     # Convert True (harmless) -> 0 and False (harmful) -> 1
#     labels = t.where(t.tensor(labels) == True, t.tensor(0), t.tensor(1))
    
#     ### Data Batches
#     # random batch sample
#     idxs = list(range(data_split.num_rows))
#     random.Random(seed).shuffle(idxs)
#     data, labels = [data[i] for i in idxs], [labels[i] for i in idxs]
   
#     batches = [
#         (data[i:i+batch_size], t.tensor(labels[i:i+batch_size], device=DEVICE)) for i in range(0, len(data), batch_size)
#     ]

#     return batches

### function used to process data into harmful (1) and nonharmful (0)
def load_target_concept_data(train=True, target_label=1): 

    dataset = load_concept_ds()

    if train:
        data_split = dataset['train']
    else:
        data_split = dataset['test']

    ### Data processing: 
    ## flatten dataset [list of sentences] + [list of labels]
    data = [item for sublist in data_split['sentence'] for item in sublist]
    labels = [item for sublist in data_split['label'] for item in sublist]

    # Convert True (harmless) -> 0 and False (harmful) -> 1
    labels = t.where(t.tensor(labels) == True, t.tensor(0), t.tensor(1))

    idxs = list(range(data_split.num_rows))
    data, labels = [data[i] for i in idxs], [labels[i] for i in idxs]
    d = (data, labels)
    target_data_text = [text for text, label in zip(d[0], d[1]) if label.item() == target_label]
    target_data_label = [label.tolist() for text, label in zip(d[0], d[1]) if label.item() == target_label]
    target_data_ds = target_data_text, target_data_label
    assert np.unique(target_data_label) == target_label

    return target_data_ds


    

