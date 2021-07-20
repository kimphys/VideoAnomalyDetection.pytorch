import torch
from torch.utils.data import DataLoader

from model import LSTMAutoEncoder
from dataset import SequenceDataset

from tqdm import tqdm

import numpy as np

class args():
    
    # Model setting
    checkpoint = 'weights/ckpt_100.pt'

    # Dataset setting
    channels = 1
    size = 256
    frames_dir = 'datasets/Test001'
    time_steps = 10

    # For GPU training
    gpu = 0 # None

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

    with torch.set_grad_enabled(False):
        pbar = tqdm(testloader)
        
        for i, seqs in enumerate(pbar):
            model.eval()

            if use_cuda: seqs = seqs.cuda()

            outs = model(seqs)

            if use_cuda:
                seqs = seqs[0].detach().cpu()
                outs = outs[0].detach().cpu()
            else:
                seqs = seqs[0]
                outs = seqs[0]

            
            # distance = (seqs - outs) * (seqs - outs)
            # distance = distance / torch.max(distance) * 255

            gts.append(seqs.numpy())
            preds.append(outs.numpy())

            seqs_reconstruction_cost = np.array([np.linalg.norm(np.subtract(gts[j],preds[j])) for j in range(0,i+1)])      
            sa = (seqs_reconstruction_cost - np.min(seqs_reconstruction_cost)) / np.max(seqs_reconstruction_cost)
            sr = 1 - sa

            if i == 0:
                regularity_score.extend(sr)
            else:
                regularity_score.append(sr[-1])

    f = open('result.csv','w')
    for i, score in enumerate(regularity_score):
        vstr = str(i) + ',' + str(score) + '\n'
        f.write(vstr)
    f.close()

if __name__ == "__main__":
    evaluate()