import argparse
import gzip
import os
import random
import warnings
from pathlib import Path

import numpy as np
import orjson as json
import pandas as pd
import torch
from more_itertools import chunked
from tqdm.auto import tqdm
from utils import (
    amazon_dataset2fullname,
    check_path,
    clean_text,
    core_n,
    load_plm,
    timeit,
)

warnings.filterwarnings('ignore')


@timeit
def preprocess_rating(args):
    ratings_df = pd.read_csv(
        os.path.join(args.input_path, 'Ratings', args.dataset_full_name + '.csv'),
        dtype={'time': int, 'rating': float},
        names=['asin', 'user_id', 'rating', 'time']
    ).dropna().drop_duplicates(subset=['asin', 'user_id'])

    # remove items that don't have generated texts in both kg_v2 and kg_v3
    meta_folder = os.path.join(args.input_path, 'Generated')
    meta_asins = set(
        pd.read_table(
            os.path.join(meta_folder, args.dataset_full_name + '_kg_v2.tsv'), usecols=['asin']
        ).asin.unique()
    ) & set(
        pd.read_table(
            os.path.join(meta_folder, args.dataset_full_name + '_kg_v3.tsv'), usecols=['asin']
        ).asin.unique()
    )

    ratings_df = ratings_df[ratings_df.asin.isin(meta_asins)]
    ratings_df = core_n(ratings_df, n=args.user_k, columns=['user_id'])
    return ratings_df.sort_values(by=['time'])


@timeit
def generate_text_f_meta(args, items, features):
    item_text_list = []
    already_items = set()
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{args.dataset_full_name}.json.gz')
    with gzip.open(meta_file_path, 'r') as fp:
        for line in fp:
            data = json.loads(line)
            item = data['asin']
            if item in items and item not in already_items:
                already_items.add(item)
                text = ''
                for meta_key in features:
                    if meta_key in data:
                        meta_value = clean_text(data[meta_key])
                        text += meta_value + ' '
                item_text_list.append([item, text])
    return item_text_list


@timeit
def generate_text_f_generated(args, items):
    path = os.path.join(args.input_path, 'Generated', args.kg_path)
    data = pd.read_table(path).dropna()
    data.columns = [i.replace('_w_reviews', '') for i in data.columns]
    data = data[data.asin.isin(items)]
    item_text_list = data.apply(lambda x: (x['asin'], ' '.join([f'{f.replace("gen_", "")}: {x[f]}' for f in args.kg_features])), axis=1).tolist()
    return item_text_list


def write_text_file(item_text_list, file):
    with open(file, 'w') as fp:
        fp.write('item_id:token\ttext:token_seq\n')
        for item, text in item_text_list:
            fp.write(item + '\t' + text + '\n')


def preprocess_text(args, items):

    if args.vanilla_features:
        item_text_list = generate_text_f_meta(args, items, args.vanilla_features)

    if args.kg_features:
        item_text_list_gen = generate_text_f_generated(args, items)

    if args.vanilla_features and args.kg_features:
        for i in range(len(item_text_list)):
            item_text_list[i][1] += ' ' + item_text_list_gen[i][1]
    elif args.kg_features:
        item_text_list = item_text_list_gen

    write_text_file(item_text_list, os.path.join(args.output_path, f'{args.dataset}.text'))
    return item_text_list


def generate_training_data(rating_df):
    # generate train valid test
    mappings = {
        'user': dict(zip(rating_df.user_id.unique(), range(rating_df.user_id.nunique()))),
        'item': dict(zip(rating_df.asin.unique(), range(rating_df.asin.nunique())))
    }

    rating_df['user_id'] = rating_df.user_id.map(mappings['user'])
    rating_df['asin'] = rating_df.asin.map(mappings['item']).astype(str)

    groupped = rating_df.groupby('user_id')
    interactions = {}
    interactions['train'] = (
        groupped.apply(lambda x: x.iloc[:-2], include_groups=False)['asin']
        .reset_index(level=1, drop=True)
        .reset_index()
        .groupby('user_id')
        .agg(list)['asin']
        .to_dict()
    )
    interactions['valid'] = (
        groupped.apply(lambda x: x.iloc[-2], include_groups=False)
        .groupby('user_id')
        .agg(list)['asin']
        .to_dict()
    )
    interactions['test'] = (
        groupped.tail(1)
        .reset_index(drop=True)
        .groupby('user_id')
        .agg(list)['asin']
        .to_dict()
    )
    assert len(interactions['train']) == len(interactions['valid']) == len(interactions['test'])
    return interactions, mappings


@timeit
def generate_item_embedding(args, item_text_list, item_map, tokenizer, model, batch_size, drop_ratio=0):

    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item_map[item]] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    for sentences in tqdm(
        chunked(order_texts, batch_size),
        total=len(order_texts) // batch_size + 1,
        desc=f'Embedding wdr={drop_ratio}',
        dynamic_ncols=True,
        disable=args.quiet,
    ):
        if drop_ratio > 0:
            random.seed(args.seed)
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences

        if args.plm_name == 'all-MiniLM-L6-v2':
            embeddings.append(model.encode(sentences))
            continue

        encoded = tokenizer(
            sentences,
            padding=True,
            max_length=args.max_length,
            truncation=True,
            return_tensors='pt',
        ).to(args.device)
        outputs = model(**encoded)

        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0,].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:, 1:, :].sum(dim=1) / encoded['attention_mask'][:, 1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu().numpy()
            embeddings.append(mean_output)

    embeddings = np.concatenate(embeddings)
    print('Embeddings shape: ', embeddings.shape, flush=True)

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    suffix = '2' if drop_ratio > 0 else '1'

    file = os.path.join(args.output_path, args.dataset + '.feat' + suffix + args.emb_type)
    embeddings.tofile(file)


def convert_to_atomic_files(args, train, valid, test, context_length=50):
    uid_list = list(train.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-context_length:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train[uid][-context_length:]
            target_item = valid[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train[uid] + valid[uid])[-context_length:]
            target_item = test[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='Scientific', help='Toys/Clothing')
    parser.add_argument('--user_k', type=int, default=3, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=0, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='../raw/')
    parser.add_argument('--output_path', type=str, default='../downstream/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=0, help='word drop ratio, do not drop by default')
    parser.add_argument('--max_length', type=int, default=512, help='max length of input text')
    parser.add_argument(
        '--kg_features',
        type=str,
        nargs='*',
        choices=['gen_description', 'gen_usecases', 'gen_expert'],
        help='KG features',
    )
    parser.add_argument('--kg_path', type=str, default='kg_v2.tsv', help='KG file path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for PLM')
    parser.add_argument('--quiet', action='store_true', help='disable tqdm')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--vanilla_features', type=str, nargs='*', help='features to use', choices=['title', 'category', 'brand', 'description'])
    args = parser.parse_args()

    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id > -1 else 'cpu')

    assert args.vanilla_features or args.kg_features, 'at least one of vanilla and kg_features should be True'
    args.dataset_full_name = amazon_dataset2fullname[args.dataset]
    if args.vanilla_features:
        args.dataset += '_w_' + '_'.join(args.vanilla_features)
    if args.kg_features:
        kg_name = '_'.join([i.replace('gen_', '') for i in args.kg_features])
        args.dataset += f"_{kg_name}_{Path(args.kg_path).stem}"

    args.kg_path = f'{args.dataset_full_name}_{args.kg_path}'
    args.output_path = os.path.join(args.output_path, 'dd', args.dataset)
    check_path(args.output_path)

    return args


def main():

    args = parse_args()

    rating_inters = preprocess_rating(args)
    item_text_list = preprocess_text(args, set(rating_inters.asin.unique()))
    interactions, mappings = generate_training_data(rating_inters)

    # device & plm initialization
    tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(args.device)

    # generate PLM emb and save to file
    generate_item_embedding(
        args,
        item_text_list,
        mappings['item'],
        tokenizer,
        plm_model,
        batch_size=args.batch_size,
        drop_ratio=0,
    )
    # pre-stored word drop PLM embs
    if args.word_drop_ratio > 0:
        generate_item_embedding(
            args,
            item_text_list,
            mappings['item'],
            tokenizer,
            plm_model,
            batch_size=args.batch_size,
            drop_ratio=args.word_drop_ratio,
        )

    # save interaction sequences into atomic files
    convert_to_atomic_files(args, **interactions)


if __name__ == '__main__':
    main()
