import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    name = name.lower()

    if name == "relu":
        return nn.ReLU(inplace=True)

    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)

    if name == "silu":
        return nn.SiLU(inplace=True)

    raise ValueError("Activation must be one of ['relu','leakyrelu','silu']")

class DoubleConv(nn.Module):
    """(convolution => activation) * 2"""
    def __init__(self, in_channels, out_channels, activation="silu"):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            get_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            get_activation(activation),
        )

    def forward(self, x):
        return self.double_conv(x)

class BilinearUpConv(nn.Module):
    """
        F.interpolate and a 1x1 Conv to reduce channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv(x)
    
class GradualExpansionUNet(nn.Module):
    def __init__(self, in_msi=12, in_hsi=230, interpolation_mode='ConvTranspose2d', activation='silu'):
        """
           interpolation_mode in ['ConvTranspose2d', 'Bilinear']. With ConvTranspose2d, 
           the upsampling is learnable (and could therefore introduce artefacts). With Bilinear, 
           the upsampling is smoother.

           activation in ['relu', 'leakyrelu', 'silu']
        """
        super().__init__()

        if interpolation_mode not in ['ConvTranspose2d', 'Bilinear']:
            raise ValueError("interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'")

        self.interpolation_mode = interpolation_mode

        # [B, 12, H, W]
        self.inc = DoubleConv(in_msi, 64, activation) # [B, 64, H, W]

        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128, activation))  # [B, 128, H/2, W/2]
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256, activation)) # [B, 256, H/4, W/4]
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512, activation)) # [B, 512, H/8, W/8]

        if self.interpolation_mode == 'ConvTranspose2d':
            self.up1a = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # [B, 256, H/4, W/4]
            self.up2a = nn.ConvTranspose2d(384, 256, kernel_size=2, stride=2) # [B, 256, H/2, W/2]
            self.up3a = nn.ConvTranspose2d(256, 192, kernel_size=2, stride=2) # [B, 192, H, W]
        else: # Bilinear
            self.up1a = BilinearUpConv(512, 256)
            self.up2a = BilinearUpConv(384, 256)
            self.up3a = BilinearUpConv(256, 192)


        # Concat with down2: [B, 256+256, H/4, W/4]
        self.up1b = DoubleConv(512, 384, activation)    # [B, 384, H/4, W/4]
        # Concat with down1: [B, 256+128, H/2, W/2]
        self.up2b = DoubleConv(384, 256, activation)    # [B, 256, H/2, W/2]
        # Concat with inc: [B, 192+64, H, W]
        self.up3b = DoubleConv(256, 230, activation)    # [B, 230, H, W]
        
        self.outc = nn.Conv2d(230, in_hsi, kernel_size=1) # [B, 230, H, W]

    def forward(self, x):
        s1 = self.inc(x)          # [B, 64, H, W]
        s2 = self.down1(s1)       # [B, 128, H/2, W/2]
        s3 = self.down2(s2)       # [B, 256, H/4, W/4]
        b = self.down3(s3)        # [B, 512, H/8, W/8]
        
        x = self.up1a(b)          # [B, 256, H/4, W/4]
        x = torch.cat([x, s3], 1) # [B, 512, H/4, W/4]
        x = self.up1b(x)          # [B, 384, H/4, W/4]
        
        x = self.up2a(x)          # [B, 256, H/2, W/2]
        x = torch.cat([x, s2], 1) # [B, 384, H/2, W/2]
        x = self.up2b(x)          # [B, 256, H/2, W/2]
        
        x = self.up3a(x)          # [B, 192, H, W]
        x = torch.cat([x, s1], 1) # [B, 256, H, W]
        x = self.up3b(x)          # [B, 230, H, W]
        
        return self.outc(x)      # [B, 230, H, W]

class UNet(nn.Module):
    def __init__(self, in_channels=230, out_channels=230, base_features=64, interpolation_mode='ConvTranspose2d', learning_mode='residual', activation='silu'):
        """
           interpolation_mode in ['ConvTranspose2d', 'Bilinear']. With ConvTranspose2d, 
           the upsampling is learnable (and could therefore introduce artefacts). With Bilinear, 
           the upsampling is smoother.

           learning_mode in ['standard', 'residual']

           activation in ['relu', 'leakyrelu', 'silu']
        """
        super().__init__()

        if interpolation_mode not in ['ConvTranspose2d', 'Bilinear']:
            raise ValueError("interpolation_mode must be 'ConvTranspose2d' or 'Bilinear'")
        
        if learning_mode not in ['standard', 'residual']:
            raise ValueError("learning_mode must be 'standard' or 'residual'")

        self.interpolation_mode = interpolation_mode
        self.learning_mode = learning_mode

        self.inc = DoubleConv(in_channels, base_features, activation)           # [B, 64, H, W]
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features, base_features*2, activation))   # [B, 128, H/2, W/2]
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features*2, base_features*4, activation)) # [B, 256, H/4, W/4]
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_features*4, base_features*8, activation)) # [B, 512, H/8, W/8]

        if self.interpolation_mode == 'ConvTranspose2d':
            self.up1a = nn.ConvTranspose2d(base_features*8, base_features*4, kernel_size=2, stride=2)  # [B, 256, H/4, W/4]
            self.up2a = nn.ConvTranspose2d(base_features*4, base_features*2, kernel_size=2, stride=2)  # [B, 128, H/2, W/2]
            self.up3a = nn.ConvTranspose2d(base_features*2, base_features, kernel_size=2, stride=2)    # [B, 64, H, W]
        else: # Bilinear
            self.up1a = BilinearUpConv(base_features*8, base_features*4)
            self.up2a = BilinearUpConv(base_features*4, base_features*2)
            self.up3a = BilinearUpConv(base_features*2, base_features)

        self.up1b = DoubleConv(base_features*8, base_features*4, activation)    
        self.up2b = DoubleConv(base_features*4, base_features*2, activation)    
        self.up3b = DoubleConv(base_features*2, base_features, activation)    

        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1) 

    def forward(self, x):
        id = x 
        
        s1 = self.inc(x)              # [B, 64, H, W]
        s2 = self.down1(s1)           # [B, 128, H/2, W/2]
        s3 = self.down2(s2)           # [B, 256, H/4, W/4]
        b = self.down3(s3)            # [B, 512, H/8, W/8] 
        
        out = self.up1a(b)            # [B, 256, H/4, W/4]
        out = torch.cat([out, s3], 1) # [B, 512, H/4, W/4]
        out = self.up1b(out)          # [B, 256, H/4, W/4]
        
        out = self.up2a(out)          # [B, 128, H/2, W/2]
        out = torch.cat([out, s2], 1) # [B, 256, H/2, W/2]
        out = self.up2b(out)          # [B, 128, H/2, W/2]
        
        out = self.up3a(out)          # [B, 64, H, W]
        out = torch.cat([out, s1], 1) # [B, 128, H, W]
        out = self.up3b(out)          # [B, 64, H, W]
        
        res = self.outc(out)  

        if self.learning_mode == 'standard':
            return res
        return id + res