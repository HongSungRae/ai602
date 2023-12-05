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
# https://github.com/pytorch/pytorch/issues/40497 # autocast NaN Loss
def train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs, save_path, args=None):
    scaler = GradScaler()
    logger_train_loss = log.Logger(f'{save_path}/train_loss.log')
    logger_train_acc = log.Logger(f'{save_path}/train_accuracy.log')
    logger_iter_gradient = log.Logger(f'{save_path}/train_gradient.log')
    logger_iter_logit = log.Logger(f'{save_path}/train_logit.log')
    iteration = 0
    count = 0
    best_acc = 0
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = log.AverageMeter()
        train_acc = log.AverageMeter()
        val_acc = log.AverageMeter()
        for idx, (x, target) in enumerate(train_dataloader):
            iteration += 1
            x, target = x.cuda(), target.cuda()
            # with autocast():
            pred = model(x)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()         
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update() 
            accucacy = torchmetrics.functional.accuracy(pred.detach().cpu(), 
                                                        target.detach().cpu(), 
                                                        'multiclass', 
                                                        num_classes=args.num_classes, 
                                                        top_k=1)
            total_loss.update(loss.item(), x.shape[0])
            train_acc.update(accucacy.item(), x.shape[0])
            # max_gradient_value = 0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         gradient_value = torch.max(torch.abs(param.grad)).item()
            #         if gradient_value > max_gradient_value:
            #             max_gradient_value = gradient_value
            # logger_iter_gradient.write([iteration, max_gradient_value])
            # logger_iter_logit.write([iteration, torch.max(pred).item()])
        scheduler.step()
        logger_train_loss.write([epoch, total_loss.avg])
        logger_train_acc.write([epoch, train_acc.avg])

        model.eval()
        for idx, (x, target) in enumerate(val_dataloader):
            x = x.cuda()
            with torch.no_grad():
                pred = model(x)
                val_accucacy = torchmetrics.functional.accuracy(pred.detach().cpu(),
                                                                target.detach().cpu(),
                                                                'multiclass',
                                                                num_classes=args.num_classes,
                                                                top_k=1)
            val_acc.update(val_accucacy, x.shape[0])

        print(f'{epoch+1} | {epochs} : {total_loss.avg:.3f} | {train_acc.avg:.3f} | {val_acc.avg:.3f}')

        if val_acc.avg > best_acc:
            best_acc = val_acc.avg
            count = 0
            torch.save(model.module.state_dict(), f'{save_path}/model.pt')
        else:
            count += 1
            if count == 15:
                break

    log.draw_curve(save_path, logger_train_loss, logger_train_loss)



def test(model, test_dataloader, num_classes, save_path, args=None):
    print('++++++++ Test를 시작합니다 ++++++++')
    model.eval()
    total_acc1 = log.AverageMeter()
    total_acc5 = log.AverageMeter()
    total_auroc = log.AverageMeter()
    get_acc_at_1 = Accuracy(task='multiclass', num_classes=num_classes)
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
    utils.save_as_json(save_path, 'results', results)
    print(results)
    print('++++++++ Test를 종료합니다 ++++++++')