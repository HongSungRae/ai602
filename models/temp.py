import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchsummary import summary
import numpy as np


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)




class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ResidualBlock(nn.Module):
  def __init__(self, in_feature):
    super().__init__()

    self.block = nn.Sequential(
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_feature, in_feature, 3),
      nn.InstanceNorm2d(in_feature),
      nn.ReLU(inplace=True),
      nn.ReflectionPad2d(1),
      nn.Conv2d(in_feature, in_feature, 3),
      nn.InstanceNorm2d(in_feature),
    )
  
  def forward(self, x):
    return x + self.block(x)
  

    


class ViTUnet(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert depth % 2 == 0, 'encoder-decoder should have same depth'
        assert depth % 3 == 0, 'downsample and upsample is consisted with 3 levels'
        assert image_size % 4 == 0 and patch_size % 4 == 0, 'downsampling size error'
        # variables
        self.dim = dim
        self.channels = channels
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # networks
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        encoder_list = []
        decoder_list = []
        for i in range(3):
            transformer_encoder = Transformer(dim=int(dim*(0.5**i)),
                                              depth=int(depth/6),
                                              heads=heads,
                                              mlp_dim=int(mlp_dim*(0.5**i)))
            transformer_decoder = Transformer(dim=int(dim*(0.5**(2-i))),
                                              depth=int(depth/6),
                                              heads=heads,
                                              mlp_dim=int(mlp_dim*(0.5**(2-i))))
            encoder_list.append(transformer_encoder)
            decoder_list.append(transformer_decoder)
        self.encoders = nn.ModuleList(encoder_list)
        self.decoders = nn.ModuleList(decoder_list)
        self.peak = Transformer(dim=int(dim/8),
                                depth=int(depth/6),
                                heads=heads,
                                mlp_dim=int(mlp_dim/8))
        # self.to_out = nn.Sequential(nn.Linear(dim*4, patch_dim))
        self.to_out = nn.Linear(dim, patch_dim)
        cnn = [nn.ReflectionPad2d(1),
               nn.Conv2d(12,64,3),
               nn.InstanceNorm2d(64),
               nn.ReLU(),
               nn.ReflectionPad2d(1),
               nn.Conv2d(64,128,3),
               nn.InstanceNorm2d(128),
               nn.ReLU()]
        for _ in range(int(depth/3)):
            cnn += [ResidualBlock(128)]
        cnn += [nn.ReflectionPad2d(1),
                nn.Conv2d(128,64,3),
                nn.InstanceNorm2d(64),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64,3,3,1)]
        self.cnn = nn.Sequential(*cnn)

        self.cnn1d = nn.Sequential(nn.Conv1d(4*self.num_patches, 8*self.num_patches, 1),
                                   nn.InstanceNorm1d(8*self.num_patches),
                                   nn.ReLU(),
                                   nn.Conv1d(8*self.num_patches, 16*self.num_patches, 1),
                                   nn.InstanceNorm1d(16*self.num_patches),
                                   nn.ReLU(),
                                   nn.Conv1d(16*self.num_patches, 8*self.num_patches, 1),
                                   nn.InstanceNorm1d(8*self.num_patches),
                                   nn.ReLU(),
                                   nn.Conv1d(8*self.num_patches, 4*self.num_patches, 1),
                                   nn.InstanceNorm1d(4*self.num_patches),
                                   nn.ReLU(),
                                   nn.Conv1d(4*self.num_patches, self.num_patches, 1))
    

    def forward(self, x, mask=None):
        # 1. Reshape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # 2. embedding
        x = self.patch_to_embedding(x)
        x += self.pos_embedding

        # 3. Encode
        encoder_features = []
        for idx, encoder in enumerate(self.encoders): # range(3)
            x = F.interpolate(x, scale_factor=[1,0.5,0.5][idx]) # down sample
            # x += F.interpolate(self.pos_embedding, scale_factor=0.5**idx)
            x = encoder(x)
            encoder_features.append(x)
        x = F.interpolate(x, scale_factor=0.5)
        # x += F.interpolate(self.pos_embedding, scale_factor=0.5**3)
        x = self.peak(x)

        # 4. Decode
        for idx, decoder in enumerate(self.decoders): # range(3)
            x = F.interpolate(x, scale_factor=2) # up sample
            x = torch.cat([encoder_features[-idx-1], x], dim=1)
            x = decoder(x)
        # x = rearrange(x, 'b (t h w) (dim) -> b (h w) (t dim)', t=4,h=int(self.num_patches**0.5))
        x = self.to_out(x) # (b, 4hw, dim)-> (b, 4hw, ppc=patch_dim)
        
        # 5. Reshape
        # x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=self.channels,p1=self.patch_size,p2=self.patch_size,h=int(self.num_patches**0.5))
        # x = rearrange(x, 'b (t h w) (p1 p2 c) -> b (t c) (h p1) (w p2)', 
        #               p1=self.patch_size, p2=self.patch_size, c=self.channels, t=4, h=int(self.num_patches**0.5))

        # 6. CNN
        # x = self.cnn(x)
        x = self.cnn1d(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (c) (h p1) (w p2)', 
                      p1=self.patch_size, p2=self.patch_size, c=self.channels, h=int(self.num_patches**0.5))
        
        # 7. tanh 
        x = torch.tanh(x)
        return x





if __name__ == '__main__':
    image_size = 224
    patch_size = 32

    vitunet = ViTUnet(image_size=image_size,
                      patch_size=patch_size,
                      dim=192*2,
                      depth=12,
                      heads=6,
                      mlp_dim=768*2)
    x = torch.zeros(16,3,image_size,image_size)
    recon_x = vitunet(x)
    
    summary(vitunet, (3,image_size,image_size), device='cpu')
    print(recon_x.shape)
    print(torch.min(recon_x), torch.max(recon_x))