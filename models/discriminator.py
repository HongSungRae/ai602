import torch.nn as nn
import torch
from torchsummary import summary


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        layer_list = []
        for i,_ in enumerate(in_channels):
            layer_list.append(self.getConv(in_channels[i],
                                           out_channels[i],
                                           kernel_size[i],
                                           stride[i],
                                           padding[i]))
        else:
            layer_list.append(nn.Conv2d(out_channels[-1],1,4,1,1))
            layer_list.append(nn.Sigmoid())
            self.layers = nn.ModuleList(layer_list)
    
    def getConv(self,in_channels,out_channels,kernel_size,stride,padding):
        layers = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
                               nn.InstanceNorm2d(out_channels),
                               nn.LeakyReLU(.2))
        return layers
    
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x



if __name__ == '__main__':
    x = torch.randn((16,3,256,256))
    discriminator = PatchDiscriminator(in_channels=[3,16,32,16],
                                       out_channels=[16,32,16,3],
                                       kernel_size=[4,4,4,4],
                                       stride=[2,2,2,2],
                                       padding=[2,2,2,2])
    y_pred = discriminator(x)
    print(f'output shape : {y_pred.shape}\n')
    print(f'(min, max) : ({torch.min(y_pred):.2f},{torch.max(y_pred):.2f})')
    summary(discriminator, (3,256,256), device='cpu')

    del x, discriminator, y_pred