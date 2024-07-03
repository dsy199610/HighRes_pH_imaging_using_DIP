import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet_T1_T2_DWI(nn.Module):

    def __init__(self, in_channels, init_features, out_channels=1):
        super(UNet_T1_T2_DWI, self).__init__()
        features = init_features
        self.encoder1 = Encoder(in_channels=in_channels, init_features=features)

        self.conv4 = nn.Conv3d(in_channels=features * 8, out_channels=features * 8, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
        self.decoder4 = BasicConv3d(features * 8 * 2, features * 8, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.conv3 = nn.Conv3d(in_channels=features * 8, out_channels=features * 4, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
        self.decoder3 = BasicConv3d(features * 4 * 2, features * 4, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(in_channels=features * 4, out_channels=features * 2, kernel_size=(3, 3, 1), stride=1, padding=(1, 1, 0))
        self.decoder2 = BasicConv3d(features * 2 * 2, features * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.conv1 = nn.Conv3d(in_channels=features * 2, out_channels=features, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.decoder1 = BasicConv3d(features * 2, features, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.conv_output = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, T1, T2, DWI, DownRes):
        enc1_1, enc1_2, enc1_3, enc1_4, bottleneck1 = self.encoder1(torch.cat((T1, T2, DWI), dim=1))

        dec4 = self.conv4(F.interpolate(bottleneck1, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True))
        dec4 = torch.cat((enc1_4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.conv3(F.interpolate(dec4, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True))
        dec3 = torch.cat((enc1_3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.conv2(F.interpolate(dec3, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True))
        dec2 = torch.cat((enc1_2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.conv1(F.interpolate(dec2, scale_factor=(2, 2, 4), mode='trilinear', align_corners=True))
        dec1 = torch.cat((enc1_1, dec1), dim=1)
        dec1 = self.decoder1(dec1)
        output_SR = 6.5 + 1.0 * torch.sigmoid(self.conv_output(dec1))
        output = F.avg_pool3d(output_SR, kernel_size=DownRes, stride=DownRes)
        return output, output_SR


class Encoder(nn.Module):

    def __init__(self, in_channels, init_features):
        super(Encoder, self).__init__()

        features = init_features
        self.encoder1 = BasicConv3d(in_channels, features, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 4), stride=(2, 2, 4))
        self.encoder2 = BasicConv3d(features, features * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.encoder3 = BasicConv3d(features * 2, features * 4, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.encoder4 = BasicConv3d(features * 4, features * 8, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        self.bottleneck = BasicConv3d(features * 8, features * 8, kernel_size=(3, 3, 1), padding=(1, 1, 0))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_pool = self.pool1(enc1)
        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        enc3 = self.encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3)
        enc4 = self.encoder4(enc3_pool)
        enc4_pool = self.pool4(enc4)
        bottleneck = self.bottleneck(enc4_pool)
        return enc1, enc2, enc3, enc4, bottleneck


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(BasicConv3d, self).__init__()
        self.padding = padding

        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding)
        self.bn = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, bias=False, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        x = F.conv3d(x, self.conv.weight, bias=None, padding=self.padding)
        x = self.bn(x)
        x = self.relu(x)

        x = F.conv3d(x, self.conv2.weight, bias=None, padding=self.padding)
        x = self.bn2(x)
        x = self.relu2(x)

        return x
