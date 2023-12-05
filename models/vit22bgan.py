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



class ViT22BGAN(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        self.transformer22B = Transformer22B(dim, depth, heads, mlp_dim)


        self.linear_decoder = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, patch_dim)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        x = self.layer_norm(x)

        x += self.pos_embedding
        x = self.transformer22B(x, mask)

        x = self.linear_decoder(x)
        print(x.shape)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p, h=int(x.shape[1]**0.5))
        x = torch.tanh(x)
        return x


if __name__ == '__main__':
    vit22b = ViT22BGAN(image_size=224,
                    patch_size=16,
                    num_classes=10,
                    dim=1024,
                    depth=24,
                    heads=16,
                    mlp_dim=4096).cuda()
    x = torch.randn((1,3,224,224)).cuda()
    y = vit22b(x)

    print(y.shape)
    print(torch.min(y), torch.max(y))

    # summary(vit22b, (3,224,224))