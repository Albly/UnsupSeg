# additional libraries were added (h5py)
# code for hdf5 files creating was added
# fast fourier transform was added (uncomment lines)
# 2 additional loss functions were added (uncomment lines)

# for using trainable parameters uncomment 30-31 lines
# for using FFT uncomment 38, 40 lines here and 49 in dataloader.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import torchaudio
from utils import LambdaLayer, PrintShapeLayer, length_to_mask, SumAlong
from dataloader import TrainTestDataset
from collections import defaultdict
from utils import (detect_peaks, max_min_norm, replicate_first_k_frames)

import h5py # for data extraction

class NextFrameClassifier(nn.Module):
    def __init__(self, hp, writefile = False):
        super(NextFrameClassifier, self).__init__()
        self.hp = hp
        self.writefile = writefile

        #Learnable parameters (TEST 2 - loss function 2)
        #self.w = nn.Parameter(torch.tensor([1.0], requires_grad= True).to('cuda'))
        #self.b = nn.Parameter(torch.tensor([0.0], requires_grad= True).to('cuda'))

        Z_DIM = hp.z_dim
        LS = hp.latent_dim if hp.latent_dim != 0 else Z_DIM

        self.enc = nn.Sequential(
            # Calculating specgram for input audio samples (TEST 3)
            #torchaudio.transforms.Spectrogram(n_fft=50, win_length=50),
            # perform summation alogng frequency axis
            #SumAlong(dim= 2),
            # CNN
            nn.Conv1d(1, LS, kernel_size=10, stride=5, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=8, stride=4, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, LS, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm1d(LS),
            nn.LeakyReLU(),
            nn.Conv1d(LS, Z_DIM, kernel_size=4, stride=2, padding=0, bias=False),
            LambdaLayer(lambda x: x.transpose(1,2)),
        )
        print("learning features from raw wav")
        
        if self.hp.z_proj != 0:
            if self.hp.z_proj_linear:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
            else:
                self.enc.add_module(
                    "z_proj",
                    nn.Sequential(
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, Z_DIM), nn.LeakyReLU(),
                        nn.Dropout2d(self.hp.z_proj_dropout),
                        nn.Linear(Z_DIM, self.hp.z_proj),
                    )
                )
                
        # # similarity estimation projections
        self.pred_steps = list(range(1 + self.hp.pred_offset, 1 + self.hp.pred_offset + self.hp.pred_steps))
        print(f"prediction steps: {self.pred_steps}")

    # 1st loss - function
    #     def score(self, f, b):
    #         return F.cosine_similarity(f, b, dim=-1) * self.hp.cosine_coef
    
    def score(self, f, b):
        c = 1/(F.cosine_similarity(f, b, dim=-1)+0.01)
        return torch.sign(c)*torch.exp(-c**4+1)
    
    # get array from cpu or convert to numpy
    def get_array(self,arr,device):
        if str(device) == 'cuda:0':
            return arr.cpu().detach().numpy()
        elif str(device) == 'cpu':
            return arr.detach().numpy()
        
    def forward(self, spect):
        # get device type
        device = spect.device
#         print(device)
        
        # input - batch of audio normalized by the longest
        # spect - torch.Size([batch size, samples])
        # wav => latent z
        z = self.enc(spect.unsqueeze(1))
        
        # if cpu => we test data on a single file
        if self.writefile == True:
            with h5py.File('data/Phonemas.hdf5', 'a') as f:
                length = len(f.keys())
                for i in range(z.shape[0]):
                    f.create_dataset('phoneme_'+str(i + length), data = self.get_array(z[i,:,:].T,device))
                f.close()

        
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):  # predict for steps 1...t
            
            pos_pred = self.score(z[:, :-t], z[:, t:])  # score for positive frame
            preds[t].append(pos_pred)
            
            for _ in range(self.hp.n_negatives):
                if self.training:
                    time_reorder = torch.randperm(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    if self.hp.batch_shuffle:
                        batch_reorder = torch.randperm(pos_pred.shape[0])
                else:
                    time_reorder = torch.arange(pos_pred.shape[1])
                    batch_reorder = torch.arange(pos_pred.shape[0])
                    
                neg_pred = self.score(z[:, :-t], z[batch_reorder][: , time_reorder])  # score for negative random frame
                preds[t].append(neg_pred)
    
        return preds

    def loss(self, preds, lengths):
        loss = 0
        for t, t_preds in preds.items():
            mask = length_to_mask(lengths - t)
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            out = out[...,0] * mask
            loss += -out.mean()
        return loss

@hydra.main(config_path='conf/config.yaml', strict=False)
def main(cfg):
    ds, _, _ = TrainTestDataset.get_datasets(cfg.timit_path)
    spect, seg, phonemes, length, fname = ds[0]
    spect = spect.unsqueeze(0)

    model = NextFrameClassifier(cfg)
    out = model(spect, length)


if __name__ == "__main__":
    main()
