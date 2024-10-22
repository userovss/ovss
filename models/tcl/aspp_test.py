import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):  
    def __init__(self, in_channels, out_channels, atrous_rates=None):  
        super(ASPP, self).__init__()
        if atrous_rates is None: 
            atrous_rates = [2, 4, 6]
            # 3*3 7 13 19
            # 2 
            # atrous_rates = [3, 5, 7, 9]

        layers = [] 
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in atrous_rates: 
            layers.append(ASPPConv(in_channels, out_channels, rate))
            '''layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, rate, padding=(rate-1)//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )) '''

        self.convs = nn.ModuleList(layers)  
        self.global_pooling = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Sequential( 
            nn.Conv2d(out_channels * (2 + len(atrous_rates)), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        '''self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )'''

    def forward(self, x):  
        x_pool = self.global_pooling(x) 
        x_pool = F.interpolate(x_pool, size=x.shape[2:], mode='bilinear', align_corners=False)  
        x_aspp = [x_pool] + [conv(x) for conv in self.convs]  
        x_cat = torch.cat(x_aspp, dim=1) 
        end_conv = self.out_conv(x_cat) 
        return end_conv

class ASPPtest(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPtest, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x_pool = self.pool(x)
        x_pool = F.interpolate(x_pool, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_pool = self.layer(x_pool)
        x = torch.concat([x, x_pool], dim=1)
        x = self.layer_2(x)
        return x
        
    

class ASPPConv(nn.Sequential): 
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True) 
        )