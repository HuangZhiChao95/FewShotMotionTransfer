import numpy as np
import torch
import torch.utils.data

class TrainSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size, filelists):
        self.filelists = filelists
        self.batch_size = batch_size
        self.indexes = []
        total = 0
        for filelist in self.filelists:
            index = np.arange(total, total+len(filelist), dtype=np.int32)
            self.indexes.append(index)
            total += len(filelist)

    def __iter__(self):
        batches = []
        for index in self.indexes:
            tmp = np.random.permutation(index)
            end = len(tmp) // self.batch_size * self.batch_size
            batches += list(tmp[:end])

        perms = np.random.permutation(np.arange(len(batches)//self.batch_size))
        for i in perms:
            yield batches[i*self.batch_size:(i+1)*self.batch_size]

    def __len__(self):
        total = 0
        for filelist in self.filelists:
            total += len(filelist) // self.batch_size
        return total


def colorize(input):

    output = input.repeat((1, 3, 1, 1))

    output[:, 0] = 0
    output[:, 1] = 1 - output[:, 1]

    return output

def d_colorize(input):
        cmap = torch.tensor([(0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153), (180, 165, 180), (150, 100, 100), (150, 120, 90),
                         (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35),
                         (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                         (0, 0, 142)], dtype=torch.float32, device=input.device)

        cmap = cmap/255
        B, _, H, W = input.size()
        cmap = cmap.transpose(0, 1).unsqueeze(1).unsqueeze(3).unsqueeze(3).expand(-1, B, -1, H, W)

        output = torch.zeros_like(input, dtype=torch.float32).repeat(1, 3, 1, 1)

        for i in range(3):
            output[:, i] = cmap[i].gather(dim=1, index=input).squeeze(1)

        return output
