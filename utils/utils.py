import os
import json


def maks_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_as_json(save_path, file_name, file):
    path = f'{save_path}/{file_name}.json'
    with open(path, 'w') as f:
        json.dump(file, f, indent=2)


def open_json(path):
    if not os.path.exists(path):
        raise ValueError(f'{path} 라는 경로를 인식할 수 없습니다.')
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file


def check_parser(args):
    assert args.experiment_name != None, '오류 : 지금 실행할 실험의 고유한 이름을 지정하세요.'
    assert args.model != None, '오류 : 학습 및 검증할 ViT 모델을 지정하세요.'
    assert args.task != None, '오류 : 어떤 Task를 수행할지 지정하세요.'
    assert args.dataset != None, '오류 : 어떤 dataset에 실험을 수행할지 지정하세요.'