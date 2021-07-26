import torch
from torch.utils.data import DataLoader

from model import LSTMAutoEncoder
from dataset import SequenceDataset

from tqdm import tqdm

import numpy as np

import imageio

from scipy import signal

class args():
    
    # Model setting
    checkpoint = 'weights/ckpt_25.pt'

    # Dataset setting
    channels = 3
    size = 256
    frames_dir = 'datasets/Test001'
    time_steps = 10

    # For GPU training
    gpu = 0 # None

def get_anomalies(i, o_seqs, seqs, outs, thres=245):

    distance = np.abs(seqs[:,0,:,:].numpy() - outs[:,0,:,:].numpy())
    dst_max = np.max(distance, axis=0)
    dst_min = np.min(distance, axis=0)
    
    regularity = dst_max - dst_min
    H = signal.convolve2d(regularity[:,:], np.ones((4,4)), mode='same')

    H = (H - np.min(H)) / (np.max(H) - np.min(H)) * 255
    H[H > thres] = 255
    H[H <= thres] = 0
    
    target = o_seqs[0][-1].permute(1,2,0).numpy().astype('uint8')

    target[H > thres,0] = 0
    target[H > thres,1] = 255
    target[H > thres,2] = 0

    return target


def evaluate():

    model = LSTMAutoEncoder(in_channels=args.channels, time_steps=args.time_steps)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available() and args.gpu is not None:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        use_cuda = False
        print('using CPU, this will be slow')

    testloader = DataLoader(dataset=SequenceDataset(channels=args.channels, size=args.size, frames_dir=args.frames_dir, time_steps=args.time_steps), batch_size=1, shuffle=False, num_workers=0)

    gts = []
    preds = []
    regularity_score = []
    anomalies = []

    with torch.set_grad_enabled(False):
        pbar = tqdm(testloader)
        
        for i, datas in enumerate(pbar):

            seqs, o_seqs = datas[0], datas[1]

            model.eval()

            if use_cuda: seqs = seqs.cuda()

            outs = model(seqs)

            if use_cuda:
                seqs = seqs[0].detach().cpu()
                outs = outs[0].detach().cpu()
            else:
                seqs = seqs[0]
                outs = outs[0]

            gts.append(seqs.numpy())
            preds.append(outs.numpy())

            anomalies.append(get_anomalies(i, o_seqs, seqs, outs, thres=245))

            seqs_reconstruction_cost = np.array([np.linalg.norm(np.subtract(gts[j],preds[j])) for j in range(0,i+1)])
            sa = (seqs_reconstruction_cost - np.min(seqs_reconstruction_cost)) / np.max(seqs_reconstruction_cost)
            sr = 1 - sa

            if i == 0:
                regularity_score.extend(sr)
            else:
                regularity_score.append(sr[-1])

    imageio.mimsave('result.gif', anomalies, fps=15) 

    f = open('result.csv','w')
    for i, score in enumerate(regularity_score):
        vstr = str(i) + ',' + str(score) + '\n'
        f.write(vstr)
    f.close()

if __name__ == "__main__":
    evaluate()