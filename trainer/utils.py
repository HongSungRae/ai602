import torch
import torch.nn as nn




# weight 초기화를 mean=0, variance=0.02
def weight_init(m): # 가중치 초기화
  classname = m.__class__.__name__

  if classname.find('Conv') != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    if hasattr(m, 'bias') and m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0.0)

  elif classname.find('BatchNorm2d') != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)




class CycleGANGeneratorLoss(nn.Module):
    def __init__(self, identity=False):
        super().__init__()
        self.lam = 10
        self.identity = identity
        self.l1 = nn.L1Loss()
    
    def forward(self,real,pred_fake,cycle,id=None):
        dis_loss = torch.mean((pred_fake-1)**2)
        cyc_loss = self.l1(real,cycle)
        loss = dis_loss + self.lam*cyc_loss
        if self.identity:
            assert id is not None
            loss += 0.5*self.lam*self.l1(real,id)
        return loss



class CycleGANDiscrimonatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred_real, pred_fake):
        loss_real = torch.mean((pred_real-1)**2)
        loss_fake = torch.mean((pred_fake)**2)
        loss = loss_real + loss_fake
        return loss




if __name__ == '__main__':
    # Variables
    A = torch.randn(16,3,256,256)
    B = torch.randn_like(A)
    pred_real_A = torch.sigmoid(torch.randn(16,1,16,16))
    pred_real_B = torch.sigmoid(torch.randn(16,1,16,16))
    pred_fake_A = torch.sigmoid(torch.randn(16,1,16,16))
    pred_fake_B = torch.sigmoid(torch.randn(16,1,16,16))
    cycle_A = torch.randn_like(A)
    cycle_B = torch.randn_like(A)
    id_A = torch.randn_like(A)
    id_B = torch.randn_like(A)

    # G
    g_criterion = CycleGANGeneratorLoss(identity=True)
    g_loss = g_criterion(A,pred_fake_A,cycle_A,id_A)
    print(f'generator loss : {g_loss:.4f}')

    # D
    d_criterion = CycleGANDiscrimonatorLoss()
    d_loss = d_criterion(pred_real_A,pred_fake_A)
    print(f'discriminator loss : {d_loss:.4f}')

    del g_criterion, d_criterion, g_loss, d_loss 