import argparse
import warnings
from collections import defaultdict
from logging import getLogger

import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import get_trainer, init_logger, init_seed, set_color

from data.dataset import UniSRecDataset
from unisrec import UniSRec

warnings.filterwarnings('ignore')


def finetune(dataset, pretrained_file, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']

    # configurations initialization
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    config["state"] = 'warning'
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # dataset filtering
    dataset = UniSRecDataset(config)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = UniSRec(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info('Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data=train_data,
        valid_data=valid_data,
        saved=True,
        show_progress=config['show_progress'],
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

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
    parser.add_argument('-f', action='store_true')
    args, unparsed = parser.parse_known_args()
    model, dataset, results = finetune(args.d, pretrained_file=args.p, fix_enc=args.f)
    print_metrics(results)
