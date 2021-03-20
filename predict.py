import argparse
import dill
from argparse import Namespace
import torch
import torchaudio
from utils import (detect_peaks, max_min_norm, replicate_first_k_frames)
from next_frame_classifier import NextFrameClassifier

import matplotlib.pyplot as plt
import h5py

def main(wav, ckpt, prominence, writefile = False):
    print(f"running inference on: {wav}")
    print(f"running inferece using ckpt: {ckpt}")
    print("\n\n", 90 * "-")

    ckpt = torch.load(ckpt, map_location=lambda storage, loc: storage)
    hp = Namespace(**dict(ckpt["hparams"]))

    # load weights and peak detection params
    model = NextFrameClassifier(hp, writefile)
    weights = ckpt["state_dict"]
    weights = {k.replace("NFC.", ""): v for k,v in weights.items()}
    model.load_state_dict(weights)
    peak_detection_params = dill.loads(ckpt['peak_detection_params'])['cpc_1']
    if prominence is not None:
        print(f"overriding prominence with {prominence}")
        peak_detection_params["prominence"] = prominence

    # load data
    audio, sr = torchaudio.load(wav)
    assert sr == 16000, "model was trained with audio sampled at 16khz, please downsample."
    audio = audio[0]
    audio = audio.unsqueeze(0)
    
    # run inference
    preds = model(audio)  # get scores
    preds = preds[1][0]  # get scores of positive pairs
    preds = replicate_first_k_frames(preds, k=0, dim=1)  # padding
    preds = 1 - max_min_norm(preds)  # normalize scores (good for visualizations)
    
#     plt.figure(figsize=(20,3))
#     plt.title('Predict function')
#     plt.plot(preds[0,:].detach().numpy())
#     plt.xlim([0,len(preds[0,:])])
    
    # premissions for file writng
    if writefile == True:
        with h5py.File('Scores.hdf5', 'a') as f:
            length = len(f.keys())
            for i in range(1):
                f.create_dataset('scores_'+str(i + length), data = preds[0,:].detach().numpy())
            f.close()
    
    preds = detect_peaks(x=preds,
                         lengths=[preds.shape[1]],
                         prominence=peak_detection_params["prominence"],
                         width=peak_detection_params["width"],
                         distance=peak_detection_params["distance"])  # run peak detection on scores
    
    preds_seconds = preds[0] * 160 / sr  # transform frame indexes to seconds
    preds_pixels = preds[0]
    
    print("predicted boundaries (in seconds):")
    print(preds_seconds)
    print("predicted boundaries (in pixels):")
    print(preds_pixels)
    print("predicted segments:")
    print(len(preds_pixels)+1)

    # premissions for file writng
    if writefile == True:
        with h5py.File('Boundaries.hdf5', 'a') as f:
            length = len(f.keys())
            for i in range(1):
                f.create_dataset('bounds_'+str(i + length), data = preds[0])
            f.close()
        
    return preds_seconds, preds_pixels, 