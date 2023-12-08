from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch
import matplotlib.pyplot as plt
import numpy as np

# local
from utils import log, utils


def train(g_AB, g_BA, d_A, d_B, g_criterion, d_criterion, g_optimizer, d_optimizer, train_dataloader, test_dataloader, epochs, save_path, identity=False, args=None):
    lr = 2e-4

    scaler_g = GradScaler()
    scaler_d = GradScaler()

    logger_g = log.Logger(f'{save_path}/train_loss_g.log')
    logger_d = log.Logger(f'{save_path}/train_loss_d.log')

    for epoch in tqdm(range(epochs)):
        g_AB.train()
        g_BA.train()
        d_A.train()
        d_B.train()
        train_g_loss = log.AverageMeter()
        train_d_loss = log.AverageMeter()
        if epoch >= 100:
            new_lr = lr*(1-(1/100)*(epoch-100))
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = new_lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = new_lr
        for i, (A, B) in enumerate(train_dataloader):
            A, B = A.cuda(), B.cuda()

            # Discriminator training
            with autocast():
                fake_B = g_AB(A)
                fake_A = g_BA(B)

                pred_real_B = d_B(B)
                pred_fake_B = d_B(fake_B)
                pred_real_A = d_A(A)
                pred_fake_A = d_A(fake_A)

                d_A_loss = d_criterion(pred_real_A, pred_fake_A)
                d_B_loss = d_criterion(pred_real_B, pred_fake_B)
                d_loss = d_A_loss + d_B_loss
            
            d_optimizer.zero_grad()
            scaler_d.scale(d_loss).backward()
            scaler_d.step(d_optimizer)
            scaler_d.update()
            train_d_loss.update(d_loss.detach().item(), A.shape[0])
            del fake_A, fake_B, pred_real_A, pred_fake_A, pred_real_B, d_A_loss, d_B_loss, d_loss
            torch.cuda.empty_cache()

            # Generator Training
            with autocast():
                fake_B = g_AB(A)
                fake_A = g_BA(B)
                cycle_B = g_AB(fake_A)
                cycle_A = g_BA(fake_B)

                pred_fake_B = d_B(fake_B)
                pred_fake_A = d_A(fake_A)

                if identity:
                    id_B = g_AB(B) # B 그대로 나와야한다
                    id_A = g_BA(A) # A 그대로 나와야한다
                else:
                    id_B = None
                    id_A = None  

                g_loss = g_criterion(A, pred_fake_A, cycle_A, id_A) +\
                         g_criterion(B, pred_fake_B, cycle_B, id_B)
            
            g_optimizer.zero_grad()
            scaler_g.scale(g_loss).backward()
            scaler_g.step(g_optimizer)
            scaler_g.update()
            train_g_loss.update(g_loss.detach().item(), A.shape[0])
            del fake_A, fake_B, cycle_A, cycle_B, pred_fake_A, pred_fake_B, g_loss, id_A, id_B
            torch.cuda.empty_cache()
        
        # log in epoch
        logger_d.write([epoch, train_d_loss.avg])
        logger_g.write([epoch, train_g_loss.avg])
        print(f'{epoch+1} | {epochs} : D({train_d_loss.avg:.3f}) | G({train_g_loss.avg:.3f})')

        if epoch+1 in [int(i*epochs*0.1) for i in range(1,10)]:
            test(g_AB, g_BA, test_dataloader, save_path, name=f'cycle_{epoch+1}')
            utils.save_model(g_AB, save_path, f'g_AB_{epoch+1}.pt', args.distributed)
            utils.save_model(g_BA, save_path, f'g_BA_{epoch+1}.pt', args.distributed)

    log.draw_curve(save_path, logger_d, logger_g, "Discriminator Loss", "Generator Loss")

            

            
    


def test(g_AB, g_BA, test_dataloader, save_path, name='cycle_final', args=None, n_imgs=5):
    print('++++++++ Test를 시작합니다 ++++++++')
    g_AB.eval()
    g_BA.eval()
    A, B = next(iter(test_dataloader))
    A, B = A.cuda(), B.cuda()
    bs = A.shape[0]

    with torch.no_grad():
        fake_B = g_AB(A)
        fake_A = g_BA(B)
        figures = torch.cat([A, fake_B, B, fake_A], dim=0)
        figures = figures*0.5 + 0.5 # [-1,1] 사이기 떄문에 [0,1]로 다시 스케일링
        figures = figures.detach().cpu().numpy()
    
    plt.figure(figsize=(30,20))
    for row in range(4):
        tag = {0:'Original A', 1:'A -> B', 2:'Original B', 3:'B -> A'}[row]
        for col in range(1,n_imgs+1):
            img = figures[row*bs+col-1]
            img = np.einsum('c...->...c', img)
            plt.subplot(4,n_imgs,row*n_imgs+col)
            plt.imshow(img)
            plt.title(tag)
    plt.savefig(f'{save_path}/{name}.png', format='png')
    print('++++++++ Test를 종료합니다 ++++++++')