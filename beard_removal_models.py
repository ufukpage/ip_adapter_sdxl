import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from arcface_resnet import iresnet18
class ResNetEncoder(nn.Module):
    def __init__(self, latent_dim, arcface_resnet_path="./backbone.pth"):
        super().__init__()
        # Load pretrained ResNet18
        if arcface_resnet_path:
            resnet = load_arcface_resnet(arcface_resnet_path)
        else:
            resnet = models.resnet18(pretrained=True)
        
        # Break down ResNet into blocks for skip connections
        self.block1 = nn.Sequential(*list(resnet.children())[:4])  # 64x128x128
        self.block2 = nn.Sequential(*list(resnet.children())[4:5])  # 64x64x64
        self.block3 = nn.Sequential(*list(resnet.children())[5:6])  # 128x32x32
        self.block4 = nn.Sequential(*list(resnet.children())[6:7])  # 256x16x16
        self.block5 = nn.Sequential(*list(resnet.children())[7:8])  # 512x8x8
        
        # Add final convolutions for mean and log variance
        self.conv_mu = nn.Conv2d(512, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(512, latent_dim, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x)      # 64x128x128
        x2 = self.block2(x1)     # 64x64x64
        x3 = self.block3(x2)     # 128x32x32
        x4 = self.block4(x3)     # 256x16x16
        x5 = self.block5(x4)     # 512x8x8
        
        mu = self.conv_mu(x5)
        logvar = self.conv_logvar(x5)
        
        return mu, logvar, [x1, x2, x3, x4, x5]

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Reshape for attention computation
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C
        k = self.key(x).view(batch_size, -1, height * width)  # B x C x HW
        v = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        
        # Compute attention scores
        attention = torch.bmm(q, k)  # B x HW x HW
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.attention = ChannelAttention(out_channels)

    def forward(self, x, skip=None):
        lat = self.lateral(x)
        if skip is not None:
            lat = F.interpolate(lat, size=skip.shape[2:], mode='nearest') + skip
        out = self.smooth(lat)
        out = self.attention(out)
        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim, attention=False):
        super().__init__()
        self.attention = attention
        
        # First decode from latent space (32x32 spatial dimension)
        self.decode1 = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.attention1 = SelfAttention(512)
        
        # Upsample to 64x64
        self.decode2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),  # 4x channels for PixelShuffle(2)
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 256x64x64
        )
        if attention:
            self.attention2 = SelfAttention(256)
        
        # Upsample to 128x128
        self.decode3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),  # 4x channels for PixelShuffle(2)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 128x128x128
        )
        if attention:
            self.attention3 = SelfAttention(128)
        
        # Upsample to 256x256
        self.decode4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # 4x channels for PixelShuffle(2)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 64x256x256
        )
        
        # Upsample to 512x512
        self.decode5 = nn.Sequential(
            nn.Conv2d(64, 12, 3, stride=1, padding=1),  # 4x channels for PixelShuffle(2)
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 3x512x512
        )
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(3, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x, skip_features):
        # Initial decoding with attention (starting at 32x32)
        x = self.decode1(x)      # 512x32x32
        if self.attention:
            x = self.attention1(x)
        x = x + skip_features[4]  # Skip connection (512x32x32)
        
        # Upsample to 64x64
        x = self.decode2(x)      # 256x64x64
        if self.attention:
            x = self.attention2(x)
        x = x + skip_features[2]  # Skip connection (256x64x64)
        
        # Upsample to 128x128
        x = self.decode3(x)      # 128x128x128
        if self.attention:
            x = self.attention3(x)
        x = x + skip_features[1]  # Skip connection (128x128x128)
        
        # Upsample to 256x256
        x = self.decode4(x)      # 64x256x256
        x = x + skip_features[0]  # Skip connection (64x256x256)
        
        # Upsample to 512x512
        x = self.decode5(x)      # 3x512x512
        
        x = self.final(x)        # 3x512x512
        return x

class DecoderFPN(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # FPN blocks
        self.fpn5 = FPNBlock(latent_dim, 512)
        self.fpn4 = FPNBlock(512, 256)
        self.fpn3 = FPNBlock(256, 128)
        self.fpn2 = FPNBlock(128, 64)
        self.fpn1 = FPNBlock(64, 32)
        
        # Decoder blocks
        self.decode1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 256x16x16
        )
        
        self.decode2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 128x32x32
        )
        
        self.decode3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 64x64x64
        )
        
        self.decode4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 32x128x128
        )
        
        self.decode5 = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.PixelShuffle(2)  # 32x256x256
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 12, 3, padding=1),
            nn.PixelShuffle(2),  # 3x512x512
            nn.Tanh()
        )

    def forward(self, x, skip_features):
        # FPN pathway (top-down)
        p5 = self.fpn5(x, skip_features[4])        # 512x8x8
        p4 = self.fpn4(p5, skip_features[3])       # 256x16x16
        p3 = self.fpn3(p4, skip_features[2])       # 128x32x32
        p2 = self.fpn2(p3, skip_features[1])       # 64x64x64
        p1 = self.fpn1(p2, skip_features[0])       # 32x128x128

        # Decoder pathway
        x = self.decode1(p5)                       # 256x16x16
        x = x + p4
        
        x = self.decode2(x)                        # 128x32x32
        x = x + p3
        
        x = self.decode3(x)                        # 64x64x64
        x = x + p2
        
        x = self.decode4(x)                        # 32x128x128
        x = x + p1
        
        x = self.decode5(x)                        # 32x256x256
        x = self.final(x)                          # 3x512x512
        
        return x


class BeardRemovalVAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = ResNetEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
        # Freeze encoder parameters
        #for param in self.encoder.parameters():
        #    param.requires_grad = False

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, skip_features = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, skip_features), mu, logvar
    

def load_arcface_resnet(weight_path):
    net = iresnet18()
    net.load_state_dict(torch.load(weight_path))
    return net