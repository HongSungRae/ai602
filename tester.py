from torch.utils.data import DataLoader
import torch
import argparse

# local 
from utils import utils
from trainer.classification import test as ctest
from models import vit, vit22b, discriminator
from dataset import cifar, apple2orange, monet2photo, imagenet


parser = argparse.ArgumentParser(description='ViT-Test')
parser.add_argument('--experiment_name', '--e', default=None, type=str,
                        help='please name your experiment')
args = parser.parse_args()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = f'./exp/{args.experiment_name}'
    configuration = utils.open_json(f'{save_path}/configuration.json')

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
        model_info = utils.open_json(f'./models/model_config.json')
        model_config = model_info[configuration['model']]
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

        ctest(model, test_dataloader, num_classes, save_path, args=None)    
    else: # gan
        pass
    



if __name__ == '__main__':
    main()