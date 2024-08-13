import numpy as np
import utils as ut

class Dataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, item):
        return self.inputs[item], self.targets[item]

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, random_state=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = ut.init_random_state(random_state)

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))
    
    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle: self.random.shuffle(indices)

        for idx in range(0, len(self.dataset), self.batch_size):
            yield self.dataset[indices[idx:idx + self.batch_size]]
