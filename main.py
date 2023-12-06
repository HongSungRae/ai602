'''
- open env
1. cmd
2. move to E:/sungrae/dtw/dtw/Scripts
3. activate

- apex ddp
https://csm-kr.tistory.com/47
https://rlawjdghek.github.io/pytorch%20&%20tensorflow%20&%20coding/DataParallel/#distributed-dataparallel-apex
https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
https://miki.tistory.com/2

- torch distributed
'''

# library
from torch.utils.data import DataLoader
import torch
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler
import time
# from apex.parallel import DistributedDataParallel as DDP
import os
import itertools
from copy import deepcopy


# local
from trainer import cyclegan, classification
from trainer.utils import CycleGANDiscrimonatorLoss, CycleGANGeneratorLoss, weight_init
from utils import utils
from models import vit, vit22b, discriminator, vitgan, vit22bgan
from dataset import cifar, apple2orange, monet2photo, imagenet


def get_parser():
    # parser 선언
    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--experiment_name', '--e', default=None, type=str,
                        help='please name your experiment')
    parser.add_argument('--model', default=None, type=str, 
                        choices=['vit-tiny', 'vit-small', 'vit-base', 'vit-large', 'vit-huge', 'vit-22bt', 'vit-22bs', 'vit-22bb', 'vit-22bl'],
                        help='model')
    parser.add_argument('--task', default=None, type=str, choices=['cls', 'classification', 'cyclegan'],
                        help='Task')
    parser.add_argument('--dataset', default=None, type=str, 
                        choices=['cifar10','cifar100','apple2orange','monet2photo','imagenet_mini', 'imagenet_tiny'],
                        help='Task')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='num classes')
    parser.add_argument('--image_size', default=224, type=int,
                        help='image width and height')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='vit image patch size')
    parser.add_argument('--batch_size', '--bs', default=32, type=int,
                        help='batch size')
    parser.add_argument('--optimizer', default='adamw', type=str,
                        help='optimizer', choices=['sgd','adamw','adagrad'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--lr_decay', default=1e-3, type=float,
                        help='learning rate decay')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight_decay') #5e-5
    parser.add_argument('--epochs', default=150, type=int,
                        help='train epoch')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed')
    parser.add_argument('--distributed', '--d', action='store_true',
                        help='False is default. How To Make True? : --distributed')
    parser.add_argument('--identity', '--i', action='store_true',
                        help='Flase is default. How To Make TRUE? : --identity')
    
    ## gpu
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id')
    parser.add_argument('--world_size', '--ws', default=2, type=int,
                        help='world size')
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    # 필수 변수 잘 입력했나 확인
    utils.check_parser(args)
    return args


def main(args):
    # init
    start = time.time()
    save_path = f'./exp/{args.experiment_name}'
    utils.maks_dir(save_path)
    utils.save_as_json(save_path, 'configuration', args.__dict__)
    utils.seed_everything(args.seed)
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # args.gpu = args.local_rank
    # torch.cuda.set_device(args.gpu)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', rank = torch.cuda.device_count(), world_size = 1)
    # torch.distributed.init_process_group(backend="nccl",  rank = torch.cuda.device_count(), world_size = 1)
    
    # dataset
    num_classes = 10
    if args.dataset in ['cifar10', 'cifar100']: # classification
        train_dataset, num_classes = cifar.get_cifar_dataset(args.dataset, 'train', args.image_size)
        test_dataset, num_classes = cifar.get_cifar_dataset(args.dataset, 'test', args.image_size)
        val_dataset = deepcopy(test_dataset)
        val_dataloader = DataLoader(val_dataset, args.batch_size, False)
    elif args.dataset == 'imagenet_tiny': # classification
        num_classes = 200
        train_dataset = imagenet.ImageNet('tiny', 'train', 64)
        test_dataset = imagenet.ImageNet('tiny', 'test', 64)
        val_dataset = imagenet.ImageNet('tiny', 'validation', 64)
        val_dataloader = DataLoader(val_dataset, args.batch_size, False)
    elif args.dataset == 'imagenet_mini': # classification
        num_classes = 1000
        train_dataset = imagenet.ImageNet('mini', 'train', 64)
        test_dataset = imagenet.ImageNet('mini', 'test', 64)
        val_dataset = deepcopy(test_dataset)
        val_dataloader = DataLoader(val_dataset, args.batch_size, False)
    elif args.dataset == 'apple2orange': # cycleGAN
        train_dataset = apple2orange.AppleOrange('train', args.image_size)
        test_dataset = apple2orange.AppleOrange('test', args.image_size)
    elif args.dataset == 'monet2photo': # cycleGAN
        train_dataset = monet2photo.Monet2Photo('train', args.image_size)
        test_dataset = monet2photo.Monet2Photo('test', args.image_size)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, args.batch_size, True)#, sampler=train_sampler, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, 16, True)#, pin_memory=True)
    args.num_classes = num_classes

    # model
    model_config = utils.open_json(r'E:/sungrae/ai602/models/model_config.json')[args.model]
    if args.task in ['cls', 'classification']:
        if '22' in args.model:
            model = vit22b.ViT22B(image_size=args.image_size,
                                  patch_size=args.patch_size,
                                  num_classes=num_classes,
                                  dim=model_config['hidden_d'],
                                  depth=model_config['layers'],
                                  heads=model_config['heads'],
                                  mlp_dim=model_config['mlp_size'])
        else:
            model = vit.ViT(image_size=args.image_size,
                            patch_size=args.patch_size,
                            num_classes=num_classes,
                            dim=model_config['hidden_d'],
                            depth=model_config['layers'],
                            heads=model_config['heads'],
                            mlp_dim=model_config['mlp_size'])
        # model = DDP(model, delay_allreduce=True)
        if args.distributed:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
    elif args.task == 'cyclegan':
        if '22' in args.model:
            g_AB = vit22bgan.ViT22BGAN(image_size=args.image_size,
                                       patch_size=args.patch_size,
                                       dim=model_config['hidden_d'],
                                       depth=model_config['layers'],
                                       heads=model_config['heads'],
                                       mlp_dim=model_config['mlp_size'])
            g_BA = vit22bgan.ViT22BGAN(image_size=args.image_size,
                                       patch_size=args.patch_size,
                                       dim=model_config['hidden_d'],
                                       depth=model_config['layers'],
                                       heads=model_config['heads'],
                                       mlp_dim=model_config['mlp_size'])
        else:
            g_AB = vitgan.ViTGAN(image_size=args.image_size,
                                 patch_size=args.patch_size,
                                 dim=model_config['hidden_d'],
                                 depth=model_config['layers'],
                                 heads=model_config['heads'],
                                 mlp_dim=model_config['mlp_size'])
            g_BA = vitgan.ViTGAN(image_size=args.image_size,
                                 patch_size=args.patch_size,
                                 dim=model_config['hidden_d'],
                                 depth=model_config['layers'],
                                 heads=model_config['heads'],
                                 mlp_dim=model_config['mlp_size'])
        d_A = discriminator.PatchDiscriminator(in_channels=[3,16,32,16],
                                               out_channels=[16,32,16,3],
                                               kernel_size=[4,4,4,4],
                                               stride=[2,2,2,2],
                                               padding=[2,2,2,2])
        d_B = discriminator.PatchDiscriminator(in_channels=[3,16,32,16],
                                               out_channels=[16,32,16,3],
                                               kernel_size=[4,4,4,4],
                                               stride=[2,2,2,2],
                                               padding=[2,2,2,2])
        g_AB.apply(weight_init)
        g_BA.apply(weight_init)
        d_A.apply(weight_init)
        d_B.apply(weight_init)
        if args.distributed:
            g_AB = torch.nn.DataParallel(g_AB).cuda()
            g_BA = torch.nn.DataParallel(g_BA).cuda()
            d_A = torch.nn.DataParallel(d_A).cuda()
            d_B = torch.nn.DataParallel(d_B).cuda()
        else:
            g_AB = g_AB.cuda()
            g_BA = g_BA.cuda()
            d_A = d_A.cuda()
            d_B = d_B.cuda()
    

    # loss
    if args.task in ['cls', 'classification']:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.task == 'cyclegan':
        g_criterion = CycleGANGeneratorLoss(args.identity).cuda()
        d_criterion = CycleGANDiscrimonatorLoss().cuda()


    # optimizer
    if args.task in ['cls', 'classification']:
        if args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        milestones = [int(args.epochs*0.3), int(args.epochs*0.6), int(args.epochs*0.9)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    elif args.task == 'cyclegan':
        lr = 2e-4
        g_optimizer = optim.Adam(itertools.chain(g_AB.parameters(), g_BA.parameters()), lr=lr, betas=[0.5, 0.999])
        d_optimizer = optim.Adam(itertools.chain(d_A.parameters(), d_B.parameters()), lr=lr, betas=[0.5, 0.999])
        

    # train -> save -> test
    if args.task in ['cls', 'classification']:
        classification.train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, args.epochs, save_path, args)
        classification.test(model, test_dataloader, num_classes, save_path, args)
    elif args.task == 'cyclegan':
        cyclegan.train(g_AB, g_BA, d_A, d_B, g_criterion, d_criterion, g_optimizer, d_optimizer, train_dataloader, args.epochs, save_path, args)
        utils.save_model(g_AB, save_path, 'g_AB.pt', args.distributed)
        utils.save_model(g_BA, save_path, 'g_BA.pt', args.distributed)
        cyclegan.test(g_AB, g_BA, test_dataloader, save_path, args=args)

    # finish
    print('\n======== 프로세스 완료 ========')
    print(f'\n모든 프로세스에 {((time.time()-start)/60):.3f}분({((time.time()-start)/3600):.3f}시간)이 소요되었습니다.')
    print(f'\n{args.experiment_name} 실험을 종료합니다.')



if __name__ == '__main__':
    args = get_parser()
    main(args)