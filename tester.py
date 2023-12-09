from torch.utils.data import DataLoader
import torch
import argparse

# local 
from utils import utils
from trainer.classification import test as clstest
from trainer.cyclegan import test as gantest
from models import vit, vit22b, vitunet, vit22bunet
from dataset import cifar, apple2orange, monet2photo, imagenet


parser = argparse.ArgumentParser(description='ViT-Test')
parser.add_argument('--experiment_name', '--e', default=None, type=str,
                        help='please name your experiment')
parser.add_argument('--distributed', '--d', action='store_true',
                    help='False is default. How To Make True? : --distributed')
args = parser.parse_args()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = f'./exp/{args.experiment_name}'
    configuration = utils.open_json(f'{save_path}/configuration.json')
    model_info = utils.open_json(f'./models/model_config.json')
    model_config = model_info[configuration['model']]

    if configuration['task'] in ['cls', 'classification']:
        # 1. test dataloader
        if configuration['dataset'] in ['cifar10', 'cifar100']:
            test_dataset, num_classes = cifar.get_cifar_dataset(configuration['dataset'], 'test', configuration['image_size'])
        elif configuration['dataset'] in ['imagenet_tiny', 'imagenet_mini']:
            num_classes = 1000
            test_dataset = imagenet.ImageNet(configuration['dataset'][9:], 'test', configuration['image_size'])
        else:
            raise ValueError("cifar와 imagenet만 classification에 구현되어있습니다.")
        test_dataloader = DataLoader(test_dataset, configuration['batch_size'], False)

        # 2. model
        if '22' in configuration['model']:
            model = vit22b.ViT22B(image_size=configuration['image_size'],
                                  patch_size=configuration['patch_size'],
                                  num_classes=num_classes,
                                  dim=model_config['hidden_d'],
                                  depth=model_config['layers'],
                                  heads=model_config['heads'],
                                  mlp_dim=model_config['mlp_size'])
        else:
            model = vit.ViT(image_size=configuration['image_size'],
                            patch_size=configuration['patch_size'],
                            num_classes=num_classes,
                            dim=model_config['hidden_d'],
                            depth=model_config['layers'],
                            heads=model_config['heads'],
                            mlp_dim=model_config['mlp_size'])
        model.load_state_dict(torch.load(f"./exp/{args.experiment_name}/model.pt"))
        if args.distributed:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
        
        # 3. test
        clstest(model, test_dataloader, num_classes, save_path, args=None) 

    else: # gan
        # 1. test dataloader
        if configuration['dataset'] == 'apple2orange':
            test_dataset = apple2orange.AppleOrange('test', 5)
        elif configuration['dataset'] == 'monet2photo':
            test_dataset = monet2photo.Monet2Photo('test', 5)
        
        # 2. model
        if configuration['model'] == 'vitunet':
            g_AB = vitunet.ViTUnet(image_size=configuration['image_size'],
                                   patch_size=configuration['patch_size'],
                                   dim=model_config['hidden_d'],
                                   depth=model_config['layers'],
                                   heads=model_config['heads'],
                                   mlp_dim=model_config['mlp_size'])
            g_BA = vitunet.ViTUnet(image_size=configuration['image_size'],
                                   patch_size=configuration['patch_size'],
                                   dim=model_config['hidden_d'],
                                   depth=model_config['layers'],
                                   heads=model_config['heads'],
                                   mlp_dim=model_config['mlp_size'])
        elif configuration['model'] == 'vit22bunet':
            g_AB = vit22bunet.ViT22BUnet(image_size=configuration['image_size'],
                                         patch_size=configuration['patch_size'],
                                         dim=model_config['hidden_d'],
                                         depth=model_config['layers'],
                                         heads=model_config['heads'],
                                         mlp_dim=model_config['mlp_size'])
            g_BA = vit22bunet.ViT22BUnet(image_size=configuration['image_size'],
                                         patch_size=configuration['patch_size'],
                                         dim=model_config['hidden_d'],
                                         depth=model_config['layers'],
                                         heads=model_config['heads'],
                                         mlp_dim=model_config['mlp_size'])
        g_AB.load_state_dict(torch.load(f'./exp/{args.experiment_name}/g_AB_final.pt'))
        g_BA.load_state_dict(torch.load(f'./exp/{args.experiment_name}/g_BA_final.pt'))
        if args.distributed:
            g_AB = torch.nn.DataParallel(g_AB).cuda()
            g_BA = torch.nn.DataParallel(g_BA).cuda()
        else:
            g_AB = g_AB.cuda()
            g_BA = g_BA.cuda()

        # 3. test
        gantest(g_AB, g_BA,
                test_dataloader,
                save_path,
                quantitative=True)
    



if __name__ == '__main__':
    main()