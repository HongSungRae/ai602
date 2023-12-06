import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math


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


class ViTUnet(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3):
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
        self.milestone = [int((2/6)*(depth/2)), int((4/6)*(depth/2)), int((6/6)*(depth/2))]

        # networks
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.scale_factor_list = []
        encoder_list = []
        decoder_list = []
        for i in range(3):
            for j in range(1,int(depth/6)+1):
                transformer_encoder = Transformer(dim=dim,
                                                  depth=int(depth/2),
                                                  heads=heads,
                                                  mlp_dim=int())
            encoder_list.append(transformer_encoder)

        for i in range(int(depth))
        self.encoders = nn.ModuleList(encoder_list)
        self.decoders = nn.ModuleList(decoder_list)
        self.peak = Transformer(dim=int(dim/4),
                                depth=int(depth/6),
                                heads=heads,
                                mlp_dim=int(mlp_dim/4))
        self.cnn = nn.Sequential(nn.Conv2d(12,64,3,1),
                                 nn.InstanceNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64,128,3,1),
                                 nn.InstanceNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128,64,3,1),
                                 nn.InstanceNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64,3,3,1))
    

    def forward(self, x, mask=None):
        # 1. Reshape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # 2. embedding
        x = self.patch_to_embedding(x)

        # 3. +=PE
        x += self.pos_embedding

        # 4. Encode
        encoder_features = []
        for idx, encoder in enumerate(self.encoders):
            if ??:
                scale_factor = ??
            x = encoder(x)
            encoder_features.append(x)
            if ??:
                x = F.interpolate(x, scale_factor=scale_factor)
        x += F.interpolate(self.pos_embedding, scale_factor=0.25)
        x = self.peak(x)

        # 5. Decode
        for idx, decoder in enumerate(self.decoders):
            scale_factor = ??
            x = F.interpolate(x, scale_factor=scale_factor)
            x = torch.cat([encoder_features[-idx], x], dim=1)
            x = decoder(x)
        
        # 4. Reshape
        x = rearrange(x, 'b (t h w) (p1 p2 c) -> b (t c) (h p1) (w p2)', 
                      p1=self.patch_size, p2=self.patch_size, c=self.channels, t=4, h=int(self.num_patches**0.5))

        # 5. CNN
        x = self.cnn(x)
        
        # 7. tanh 
        x = torch.tanh(x)
        return x

class ViTGAN(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_dim = patch_dim

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.linear_decoder = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, patch_dim)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        # print(x.shape)

        x += self.pos_embedding
        x = self.transformer(x, mask)
        # print(x.shape)

        x = self.linear_decoder(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=int(x.shape[1]**0.5))
        x = torch.tanh(x)
        return x
    

    def forward(self, img, mask=None):
        pass
        # 1. 패치 리어레인지
        # 2. 패치 임베딩
        # 3. PE 더하기
        # 4. 인코더 블럭
            # for depth/2 동안
                # depth/6 동안은 img_size, patch_size
                # 1/2다운샘플링
                # depth/6 동안은 img_size/2, patch_size/2
                # 1/2다운샘플링
                # depth/6 동안은 img_size/2, patch_szie/2
                # 1/2
        # 5. 디코더 블럭
        # 6. 리어레인지
        # 7. CNN 블럭





if __name__ == '__main__':
    pass