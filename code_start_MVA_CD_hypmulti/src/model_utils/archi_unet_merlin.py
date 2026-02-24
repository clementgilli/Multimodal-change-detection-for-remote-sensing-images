# General

# Imaging
import torch

# Local


class UNetMERLIN(torch.nn.Module):

    def __init__(self, nb_in_channels=1, nb_out_channels=1):
        super().__init__()

        self.nc_in = nb_in_channels
        self.nc_out = nb_out_channels

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upscale2d = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.leaky = torch.nn.LeakyReLU(0.1)

        self.enc0 = torch.nn.Conv2d(in_channels=self.nc_in, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc1 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc2 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc3 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc4 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc5 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.enc6 = torch.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')

        self.dec5 = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec5b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec4 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec4b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec3 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec3b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec2 = torch.nn.Conv2d(in_channels=144, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                    padding='same')
        self.dec2b = torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1a = torch.nn.Conv2d(in_channels=96 + self.nc_in, out_channels=64, kernel_size=(3, 3),
                                     stride=(1, 1), padding='same')
        self.dec1b = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')
        self.dec1 = torch.nn.Conv2d(in_channels=32, out_channels=self.nc_out, kernel_size=(3, 3), stride=(1, 1),
                                     padding='same')

    def forward(self, x):
        skips = [x]
        n = x

        n = self.leaky(self.enc0(n))
        n = self.leaky(self.enc1(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc2(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc3(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc4(n))
        n = self.pool(n)
        skips.append(n)

        n = self.leaky(self.enc5(n))
        n = self.pool(n)
        n = self.leaky(self.enc6(n))

        # DECODER
        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec5(n))
        n = self.leaky(self.dec5b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec4(n))
        n = self.leaky(self.dec4b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec3(n))
        n = self.leaky(self.dec3b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec2(n))
        n = self.leaky(self.dec2b(n))

        n = self.upscale2d(n)
        n = torch.cat((n, skips.pop()), dim=1)
        n = self.leaky(self.dec1a(n))
        n = self.leaky(self.dec1b(n))

        n = self.dec1(n)

        return n
