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
from apex.parallel import DistributedDataParallel as DDP
import os


# local
from trainer import gan, classification
from utils import utils
from models import vit, vit22b, discriminator
from dataset import cifar, apple2orange, monet2photo, imagenet


def get_parser():
    # parser 선언
    parser = argparse.ArgumentParser(description='ETRI_DoRaeMiSol')
    parser.add_argument('--experiment_name', '--e', default=None, type=str,
                        help='please name your experiment')
    parser.add_argument('--model', default=None, type=str, 
                        choices=['vit-tiny', 'vit-small', 'vit-base', 'vit-large', 'vit-huge', 'vit-22bt', 'vit-22bs', 'vit-22bb', 'vit-22bl'],
                        help='model')
    parser.add_argument('--task', default=None, type=str, choices=['cls', 'classification', 'gan'],
                        help='Task')
    parser.add_argument('--dataset', default=None, type=str, choices=['cifar10','cifar100','apple2orange','monet2photo'],
                        help='Task')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='num classes')
    parser.add_argument('--image_size', default=224, type=int,
                        help='image width and height')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='vit image patch size')
    parser.add_argument('--batch_size', '--bs', default=32, type=int,
                        help='batch size')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='optimizer', choices=['sgd','adam','adagrad'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--lr_decay', default=1e-3, type=float,
                        help='learning rate decay')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='weight_decay')
    parser.add_argument('--epochs', default=100, type=int,
                        help='train epoch')
    
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
    elif args.dataset in ['imagenet_tiny', 'imagenet_mini']: # classification
        num_classes = 1000
        train_dataset = imagenet.ImageNet(args.dataset[9:], 'train', args.image_size)
        train_dataset = imagenet.ImageNet(args.dataset[9:], 'test', args.image_size)
    elif args.dataset == 'apple2orange': # cycleGAN
        train_dataset = apple2orange.AppleOrange('train', args.image_size)
        test_dataset = apple2orange.AppleOrange('test', args.image_size)
    elif args.dataset == 'monet2photo': # cycleGAN
        train_dataset = monet2photo.Monet2Photo('train', args.image_size)
        test_dataset = monet2photo.Monet2Photo('test', args.image_size)
    elif args.dataset == 'celebA': # cycleGAN
        pass
    elif args.dataset == 'ade20k': # conditional GAN
        pass
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, args.batch_size, True)#, sampler=train_sampler, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, True)#, pin_memory=True)
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
    elif args.task == 'cyclegan':
        pass
    elif args.task == 'cgan':
        pass
    # model = DDP(model, delay_allreduce=True)
    model = torch.nn.DataParallel(model).cuda()

    # discriminator for generation task
    if args.task == 'gan':
        d = discriminator.PatchDiscriminator(in_channels=[3,16,32,16],
                                             out_channels=[16,32,16,3],
                                             kernel_size=[4,4,4,4],
                                             stride=[2,2,2,2],
                                             padding=[2,2,2,2])
        # d = DDP(d, delay_allreduce=True)
        d = torch.nn.DataParallel(d).cuda()

    # loss
    if args.task in ['cls', 'classification']:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.task == 'gan':
        pass

    # optimizer
    if args.task in ['cls', 'classification']:
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        milestones = [int(args.epochs*0.3), int(args.epochs*0.6), int(args.epochs*0.9)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
    elif args.task == 'gan':
        pass

    # train -> save -> test
    if args.task in ['cls', 'classification']:
        classification.train(model, train_dataloader, criterion, optimizer, scheduler, args.epochs, save_path, args)
        torch.save(model.module.state_dict(), f'{save_path}/model.pt')
        classification.test(model, test_dataloader, num_classes, save_path, args)
    elif args.task == 'gan':
        gan.train(model, d, train_dataloader, args.epochs)
        torch.save(model.module.state_dict(), f'{save_path}/model.pt')
        gan.test()

    # finish
    print('\n======== 프로세스 완료 ========')
    print(f'\n모든 프로세스에 {((time.time()-start)/60):.3f}분({((time.time()-start)/3600):.3f}시간)이 소요되었습니다.')
    print(f'\n{args.experiment_name} 실험을 종료합니다.')



if __name__ == '__main__':
    args = get_parser()
    main(args)