import torch
import torch.nn as nn
import numpy as np

from modules import TimeDistributed, ConvLSTM

class LSTMAutoEncoder(nn.Module):
    def __init__(self, in_channels=3,time_steps=10):
        super(LSTMAutoEncoder, self).__init__()
        ## Encoder ##
        self.conv_1 = TimeDistributed(nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(11,11), stride=4, padding=4), time_steps=time_steps)
        self.conv_2 = TimeDistributed(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5,5), stride=2, padding=2), time_steps=time_steps)

        ## LSTM Bottleneck ##
        self.lstm_1 = ConvLSTM(
            input_dim=64,
            hidden_dim=[64],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        self.lstm_2 = ConvLSTM(
            input_dim=64,
            hidden_dim=[32],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        self.lstm_3 = ConvLSTM(
            input_dim=32,
            hidden_dim=[64],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )

        ## Decoder ##
        self.deconv_1 = TimeDistributed(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=2, padding=1), time_steps=time_steps)
        self.deconv_2 = TimeDistributed(nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(10,10), stride=4, padding=3), time_steps=time_steps)
        self.conv_3 = TimeDistributed(nn.Conv2d(in_channels=128, out_channels=in_channels, kernel_size=(11,11), stride=1, padding=5), time_steps=time_steps)

    def forward(self, x):

        x = self.conv_1(x)
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        x = self.conv_2(x)
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)

        x, _ = self.lstm_1(x)
        x = x[0]
        x, _ = self.lstm_2(x)
        x = x[0]
        x, _ = self.lstm_3(x)
        x = x[0]

        x = self.deconv_1(x)
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        x = self.deconv_2(x)
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)
        x = self.conv_3(x)
        x = nn.LayerNorm(x.size()[1:], elementwise_affine=False)(x)

        return x

if __name__ == "__main__":
    input = np.random.randn(2,4,3,256,256).astype(np.float32)
    input_tensor = torch.from_numpy(input).cuda()
    model = LSTMAutoEncoder(time_steps=4).cuda()

    output_tensor = model(input_tensor)

    print(output_tensor.shape)