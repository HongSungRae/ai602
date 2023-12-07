from torchsummary import summary
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange




class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x



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



class ParallelAttentionFeed(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        assert dim%heads == 0
        super().__init__()
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim = dim
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim*3, bias=False)
        self.parallel = nn.Sequential(nn.Linear(dim, mlp_dim),
                                      nn.GELU(),
                                      nn.Linear(mlp_dim, dim))

        self.layer_norm = nn.LayerNorm(int(dim/heads), elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.to_out = nn.Linear(dim, dim)
    def forward(self, x, mask = None):
        # q,k,v,mlp_in 나누기 및 LN
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)
        # print(q.shape, k.shape, v.shape, l.shape)
        q, k = self.layer_norm(q), self.layer_norm(k)

        # q,k self attention 및 scaling
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        attn = dots.softmax(dim=-1)

        # attention으로 Value return
        attn_v = torch.einsum('bhij,bhjd->bhid', attn, v)
        attn_v = rearrange(attn_v, 'b h n d -> b n (h d)')
        attn_out = self.to_out(attn_v)

        # parallel MLP
        parallel = self.parallel(x)

        # self attention 값과 mlp 값 합쳐서 하나의 출력 만들기
        out =  attn_out + parallel
        out = self.layer_norm2(out)
        return out



class Transformer22B(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []#nn.ModuleList([])
        for _ in range(depth):
            layers.append(Residual(ParallelAttentionFeed(dim=dim, heads=heads, mlp_dim=mlp_dim)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask=None):
        for attn in self.layers:
            # print(attn)
            x = attn(x)#attn(x, mask=mask)
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
  


class ViT22BUnet(nn.Module):
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
            transformer_encoder = Transformer22B(dim=int(dim*(0.5**i)),
                                              depth=int(depth/6),
                                              heads=heads,
                                              mlp_dim=int(mlp_dim*(0.5**i)))
            transformer_decoder = Transformer22B(dim=int(dim*(0.5**(2-i))),
                                              depth=int(depth/6),
                                              heads=heads,
                                              mlp_dim=int(mlp_dim*(0.5**(2-i))))
            encoder_list.append(transformer_encoder)
            decoder_list.append(transformer_decoder)
        self.encoders = nn.ModuleList(encoder_list)
        self.decoders = nn.ModuleList(decoder_list)
        self.peak = Transformer22B(dim=int(dim/8),
                                depth=int(depth/6),
                                heads=heads,
                                mlp_dim=int(mlp_dim/8))
        self.to_out = nn.Linear(dim, patch_dim)
        cnn = [nn.ReflectionPad2d(1),
               nn.Conv2d(12,64,3),
               nn.InstanceNorm2d(64),
               nn.ReLU(),
               nn.ReflectionPad2d(1),
               nn.Conv2d(64,128,3),
               nn.InstanceNorm2d(128),
               nn.ReLU()]
        for _ in range(int(depth/6)):
            cnn += [ResidualBlock(128)]
        cnn += [nn.ReflectionPad2d(1),
                nn.Conv2d(128,64,3),
                nn.InstanceNorm2d(64),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64,3,3,1)]
        self.cnn = nn.Sequential(*cnn)
    

    def forward(self, x, mask=None):
        # 1. Reshape
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # 2. embedding
        x = self.patch_to_embedding(x)

        # 3. Encode
        encoder_features = []
        for idx, encoder in enumerate(self.encoders): # range(3)
            x = F.interpolate(x, scale_factor=[1,0.5,0.5][idx]) # down sample
            x += F.interpolate(self.pos_embedding, scale_factor=0.5**idx)
            x = encoder(x)
            encoder_features.append(x)
        x = F.interpolate(x, scale_factor=0.5)
        x += F.interpolate(self.pos_embedding, scale_factor=0.5**3)
        x = self.peak(x)

        # 4. Decode
        for idx, decoder in enumerate(self.decoders): # range(3)
            x = F.interpolate(x, scale_factor=2) # up sample
            x = torch.cat([encoder_features[-idx-1], x], dim=1)
            x = decoder(x)
        x = self.to_out(x)
        
        # 5. Reshape
        x = rearrange(x, 'b (t h w) (p1 p2 c) -> b (t c) (h p1) (w p2)', 
                      p1=self.patch_size, p2=self.patch_size, c=self.channels, t=4, h=int(self.num_patches**0.5))

        # 6. CNN
        x = self.cnn(x)
        
        # 7. tanh 
        x = torch.tanh(x)
        return x


if __name__ == '__main__':
    vit22b = ViT22BUnet(image_size=224,
                    patch_size=32,
                    dim=1024,
                    depth=24,
                    heads=16,
                    mlp_dim=4096).cuda()
    x = torch.randn((1,3,224,224)).cuda()
    y = vit22b(x)

    print(y.shape)

    summary(vit22b, (3,224,224))