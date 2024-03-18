import argparse
import os
import warnings
from collections import defaultdict

import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import get_trainer, init_seed

from data.dataset import UniSRecDataset
from unisrec import UniSRec

warnings.filterwarnings('ignore')


def get_config(args):
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    config = Config(model=UniSRec, dataset=args.d, config_file_list=props)
    config['data_path'] = os.path.join(
        os.path.dirname(config['data_path']),
        args.plm.split('/')[-1],
        os.path.basename(config['data_path'])
    )
    init_seed(config['seed'], config['reproducibility'])
    # config["state"] = 'warning'
    config['train_batch_size'] = args.batch_size
    config['eval_batch_size'] = args.batch_size
    config['gpu_id'] = args.gpu
    return config


def finetune(config, pretrained_file, fix_enc=True):
    dataset = UniSRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = UniSRec(config, train_data.dataset).to(config['device'])

    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data=train_data,
        valid_data=valid_data,
        saved=True,
        show_progress=config['show_progress'],
    )
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    return (
        config['model'],
        config['dataset'],
        {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result,
        },
    )


def print_metrics(results):
    res_val = defaultdict(lambda: defaultdict(list))
    for metric, value in results['best_valid_result'].items():
        res_val[metric.split('@')[0]][metric.split('@')[1]] = value

    res_test = defaultdict(lambda: defaultdict(list))
    for metric, value in results['test_result'].items():
        res_test[metric.split('@')[0]]['test' + metric.split('@')[1]] = value

    print(f'{"":<10}' + 'valid' + ' ' * 9 + 'test')
    print(f'{"":<10}' + 2 * ''.join([f'@{i:<6}' for i in res_val[next(iter(res_val))].keys()]))
    for metric in res_val:
        print(f'{metric:<10}', end='')
        for k in res_val[metric]:
            print(f'{res_val[metric][k]:.4f}', end=' ')
        for k in res_test[metric]:
            print(f'{res_test[metric][k]:.4f}', end=' ')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='Scientific', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('--plm', type=str, default='bert-base-uncased')  # WhereIsAI/UAE-Large-V1 all-MiniLM-L6-v2 meta-llama/Llama-2-7b-chat-hf
    parser.add_argument('-f', action='store_true')
    parser.add_argument('--gpu', '--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=2048)
    args, unparsed = parser.parse_known_args()

    config = get_config(args)
    model, dataset, results = finetune(config=config, pretrained_file=args.p, fix_enc=args.f)
    print_metrics(results)
