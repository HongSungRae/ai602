from tqdm import tqdm
from torchmetrics import Accuracy, AUROC
import torchmetrics
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import torch
import warnings
warnings.filterwarnings('ignore')

# local
from utils import log, utils


# https://computing-jhson.tistory.com/37 # amp
# https://aimaster.tistory.com/83 # amp
# https://stackoverflow.com/questions/71315491/torchmetric-calculate-accuracy-with-threshold # torchmetric acc threshold
def train(model, train_dataloader, criterion, optimizer, scheduler, epochs, save_path, args=None):
    model.train()
    scaler = GradScaler()
    logger = log.Logger(f'{save_path}/train_loss.log')
    for epoch in tqdm(range(epochs)):
        total_loss = log.AverageMeter()
        train_acc = log.AverageMeter()
        for idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()
            with autocast():
                pred = model(x)
                loss = criterion(pred, target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss.update(loss.item(), x.shape[0])
            train_acc.update(torchmetrics.functional.accuracy(pred.detach().cpu(), 
                                                              target.detach().cpu(), 
                                                              'multiclass', 
                                                              num_classes=args.num_classes, 
                                                              top_k=1),
                             x.shape[0])
        scheduler.step()
        print(f'{epoch+1} | {epochs} : {total_loss.avg:.3f} | {train_acc.avg:.3f}')
        logger.write([epoch, total_loss.avg])
    utils.save_as_json(save_path, 'train_acc.json', {'Accuracy@1':f"{total_loss.avg:.3f}"}) # 마지막 에폭만
    log.draw_curve(save_path, logger, logger)



def test(model, test_dataloader, num_classes, save_path, args=None):
    print('++++++++ Test를 시작합니다 ++++++++')
    model.eval()
    total_acc1 = log.AverageMeter()
    total_acc5 = log.AverageMeter()
    total_auroc = log.AverageMeter()
    get_acc_at_1 = Accuracy(task='multiclass', num_classes=num_classes, tok_k=1)
    get_acc_at_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
    get_auroc = AUROC(task='multiclass', num_classes=num_classes)
    with torch.no_grad():
        for idx, (x, target) in tqdm(enumerate(test_dataloader)):
            x = x.cuda()
            with autocast():
                pred = model(x)
                pred = torch.softmax(pred,dim=-1)
            pred = pred.detach().cpu()
            target = target.detach().cpu()
            acc_at_1 = get_acc_at_1(pred, target)
            acc_at_5 = get_acc_at_5(pred, target)
            auroc = get_auroc(pred, target)
            total_acc1.update(acc_at_1, x.shape[0])
            total_acc5.update(acc_at_5, x.shape[0])
            total_auroc.update(auroc, x.shape[0])
    results = {'Accuracy@1':f'{total_acc1.avg:.3f}',
               'Accuracy@5':f'{total_acc5.avg:.3f}',
               'AUROC':f'{total_auroc.avg:.3f}'}
    utils.save_as_json(save_path, 'results.json', results)
    print(results)
    print('++++++++ Test를 종료합니다 ++++++++')