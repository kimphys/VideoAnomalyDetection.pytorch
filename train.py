import os
import sys

import torch
from torch.utils.data import DataLoader

from model import LSTMAutoEncoder
from dataset import SequenceDataset

from torch.nn import MSELoss

from tqdm import tqdm

class args():
    
    # training args
    epochs = 100 # "number of training epochs, default is 2"
    save_per_epoch = 5
    batch_size = 4 # "batch size for training/testing, default is 4"
    pretrained = False
    lr_init = 1e-4
    lr_weight_decay = 1e-5
    save_model_dir = "./weights/" #"path to folder where trained model with checkpoints will be saved."
    num_workers = 1

    # Dataset setting
    channels = 3
    size = 256
    frames_dir = 'D:\\project\\convlstm\\UCSD_Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Train\\210716'
    time_steps = 10

    # For GPU training
    gpu = 0 # None

def train():

    model = LSTMAutoEncoder(in_channels=args.channels, time_steps=args.time_steps)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, betas=(0.9, 0.999), eps=1e-06, weight_decay=args.lr_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if torch.cuda.is_available() and args.gpu is not None:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        use_cuda = False
        print('using CPU, this will be slow')
        
    epoch = 0

    trainloader = DataLoader(dataset=SequenceDataset(channels=args.channels, size=args.size, frames_dir=args.frames_dir, time_steps=args.time_steps), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with torch.set_grad_enabled(True):
        for ep in range(args.epochs):

            pbar = tqdm(trainloader)
            idx = 0

            loss_sum = 0

            for i, seqs in enumerate(pbar):

                model.train()
                    
                if use_cuda: seqs = seqs.cuda()

                outs = model(seqs)

                loss = MSELoss()(seqs, outs)

                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

            print("Epoch: {}/{}, GPU: {}, Average loss: {}:".format(ep, args.epochs, torch.cuda.current_device(), loss_sum / len(pbar)))

            scheduler.step()

            if (ep + 1) % args.save_per_epoch == 0:
                # Save model
                torch.save({
                            'epoch': ep,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                        }, os.path.join(args.save_model_dir, 'ckpt_{}.pt'.format(ep + 1)))
        
    print('Finished training')

if __name__ == "__main__":
    train()