import torch
import torch.nn as nn
import torch.nn.functional as F
from src.PartialConv2d import PartialConv2d
from src.SFFA import SFFA_Module

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


# miniG1
class InpaintGenerator1(BaseNetwork):  # 128*128
    def __init__(self, init_weights=True):
        super(InpaintGenerator1, self).__init__()

        self.encoder1 = PartialConv2d(in_channels=3, out_channels=32, kernel_size=5, padding=0, multi_channel=False)
        self.encoder2 = PartialConv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=0, multi_channel=False)
        self.encoder3 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0, multi_channel=False)

        self.res_block1=ResPconvBlock(128)
        self.res_block2=ResPconvBlock(128)
        self.res_block3=ResPconvBlock(128)
        self.res_block4=ResPconvBlock(128)
        self.res_block5=ResConvBlock(128, 1)
        self.res_block6=ResConvBlock(128, 1)
        self.res_block7=ResConvBlock(128, 1)
        self.res_block8=ResConvBlock(128, 1)

        self.decoder3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)
        self.decoder2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2, padding=0)
        self.decoder1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=0)


        self.insNorm_32 = nn.InstanceNorm2d(32, track_running_stats=False)
        self.insNorm_64 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.insNorm_128 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.refPad2 = nn.ReflectionPad2d(2)

        #self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

        if init_weights:
            self.init_weights()

    def forward(self, x_in, m_in):
        ################### x, m = self.encoder(x, m)###########################
        m = m_in
        x_enc1, m = self.encoder1(self.refPad2(x_in), self.refPad2(m))
        x_enc1 = F.relu(self.insNorm_32(x_enc1), inplace=True)
        x_enc2, m = self.encoder2(x_enc1, m)
        x_enc2 = F.relu(self.insNorm_64(x_enc2), inplace=True)
        x_enc3, m = self.encoder3(x_enc2, m)
        x_enc3 = F.relu(self.insNorm_128(x_enc3), inplace=True)

        ##################### x = self.middle(x)################################
        x_mid, m = self.res_block1(x_enc3, m)
        x_mid, m = self.res_block2(x_mid, m)
        x_mid, m = self.res_block3(x_mid, m)
        x_mid, _ = self.res_block4(x_mid, m)
        x_mid = self.res_block5(x_mid)
        x_mid = self.res_block6(x_mid)
        x_mid = self.res_block7(x_mid)
        x_mid = self.res_block8(x_mid)

        ################## x = self.decoder(x)###################################

        x_dec3 = F.relu(self.insNorm_64(self.decoder3(x_mid)))
        x_dec2 = F.relu(self.insNorm_32(self.decoder2(x_dec3)))
        x_dec1 = self.decoder1(self.refPad2((x_dec2)))
        x_out = (torch.tanh(x_dec1) + 1) / 2
        # x_out = self.upsample(x_out)
        return x_out


class InpaintGenerator2(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator2, self).__init__()

        #self.encoder1 = PartialConv2d(in_channels=4, out_channels=64, kernel_size=7, padding=3, multi_channel=False)
        self.encoder1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0)
        self.encoder2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.encoder3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)


        self.res_block1=ResConvBlock(256, 2)
        self.res_block2=ResConvBlock(256, 2)
        self.res_block3=ResConvBlock(256, 2)
        self.res_block4=ResConvBlock(256, 2)
        #self.attentionBlock = AttentionModule(256)  # v4.1.4
        self.attentionBlock = SFFA_Module(256)
        self.res_block5=ResConvBlock(256, 2)
        self.res_block6=ResConvBlock(256, 2)
        self.res_block7=ResConvBlock(256, 2)
        self.res_block8=ResConvBlock(256, 2)

        # without skip connection | with skip(add) connection
        self.decoder3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.decoder2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.decoder1 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        self.refPad3 = nn.ReflectionPad2d(3)

        self.insNorm_64 = nn.InstanceNorm2d(64, track_running_stats=False)
        self.insNorm_128 = nn.InstanceNorm2d(128, track_running_stats=False)
        self.insNorm_256 = nn.InstanceNorm2d(256, track_running_stats=False)
        self.refPad3 = nn.ReflectionPad2d(3)

        if init_weights:
            self.init_weights()

    def forward(self, x_in, m_in):
        ################### x, m = self.encoder(x, m)###########################
        x_enc1 = self.encoder1(self.refPad3(x_in))
        x_enc1 = F.relu(self.insNorm_64(x_enc1), inplace=True)
        x_enc2 = self.encoder2(x_enc1)
        x_enc2 = F.relu(self.insNorm_128(x_enc2), inplace=True)
        x_enc3 = self.encoder3(x_enc2)
        x_enc3 = F.relu(self.insNorm_256(x_enc3), inplace=True)

        ##################### x = self.middle(x)################################
        x_mid = self.res_block1(x_enc3)
        x_mid = self.res_block2(x_mid)
        x_mid = self.res_block3(x_mid)
        x_mid = self.res_block4(x_mid)
        x_mid = self.attentionBlock(x_mid, F.max_pool2d(F.max_pool2d(m_in,2),2))  # 4.1.1
        x_mid = self.res_block5(x_mid)
        x_mid = self.res_block6(x_mid)
        x_mid = self.res_block7(x_mid)
        x_mid = self.res_block8(x_mid)

        ################## x = self.decoder(x)###################################

        x_dec3 = F.leaky_relu(self.insNorm_128(self.decoder3(x_mid)), 0.1)
        x_dec2 = F.leaky_relu(self.insNorm_64(self.decoder2(x_dec3)), 0.1)
        x_dec1 = self.decoder1(self.refPad3((x_dec2)))

        x_res = torch.tanh(x_dec1)
        return x_in + x_res


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResPconvBlock(nn.Module):  # dilation = 1 for inp1
    def __init__(self, dim, use_spectral_norm=False):
        super(ResPconvBlock, self).__init__()
        self.pconv1 = PartialConv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,multi_channel=False, bias=not use_spectral_norm)
        self.pconv2 = PartialConv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,multi_channel=False, bias=not use_spectral_norm)
        spectral_norm(self.pconv1, use_spectral_norm)
        spectral_norm(self.pconv2, use_spectral_norm)
        self.insNorm = nn.InstanceNorm2d(dim, track_running_stats=False)
        self.reflection_pad1 = nn.ReflectionPad2d(1)

    def forward(self, x, m):
        x1 = self.reflection_pad1(x)
        m = self.reflection_pad1(m)
        x1, m = self.pconv1(x1, m)
        x1 = F.relu(self.insNorm(x1), inplace=True)
        x1 = self.reflection_pad1(x1)
        m = self.reflection_pad1(m)
        x1, m = self.pconv2(x1, m)
        x1 = self.insNorm(x1)          # Remove ReLU at the end of the residual block   http://torch.ch/blog/2016/02/04/resnets.html
        return x + x1, m


class ResConvBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm)
        spectral_norm(self.conv1, use_spectral_norm)
        spectral_norm(self.conv2, use_spectral_norm)
        self.insNorm = nn.InstanceNorm2d(dim, track_running_stats=False)
        self.reflection_pad2 = nn.ReflectionPad2d(dilation) # 2
        self.reflection_pad1 = nn.ReflectionPad2d(1)

    def forward(self, x):
        x1 = self.reflection_pad2(x)
        x1 = self.conv1(x1)
        x1 = F.leaky_relu(self.insNorm(x1), negative_slope=0.1, inplace=True)
        x1 = self.reflection_pad1(x1)
        x1 = self.conv2(x1)
        x1 = self.insNorm(x1)          # Remove ReLU at the end of the residual block   http://torch.ch/blog/2016/02/04/resnets.html
        return x + x1


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

