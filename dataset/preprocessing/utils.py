import html
import os
import re
import string
import time
from functools import wraps
from unicodedata import normalize

import numpy as np
import torch
from bs4 import BeautifulSoup
from more_itertools import chunked
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from unidecode import unidecode

unprintable_pattern = re.compile(f'[^{re.escape(string.printable)}]')
emoji_pattern = re.compile(
    u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0"
    u"\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
    flags=re.UNICODE,
)

amazon_dataset2fullname = {
    'Beauty': 'All_Beauty',
    'Fashion': 'AMAZON_FASHION',
    'Appliances': 'Appliances',
    'Arts': 'Arts_Crafts_and_Sewing',
    'Automotive': 'Automotive',
    'Books': 'Books',
    'CDs': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Clothing': 'Clothing_Shoes_and_Jewelry',
    'Music': 'Digital_Music',
    'Electronics': 'Electronics',
    'Gift': 'Gift_Cards',
    'Food': 'Grocery_and_Gourmet_Food',
    'Home': 'Home_and_Kitchen',
    'Scientific': 'Industrial_and_Scientific',
    'Kindle': 'Kindle_Store',
    'Luxury': 'Luxury_Beauty',
    'Magazine': 'Magazine_Subscriptions',
    'Movies': 'Movies_and_TV',
    'Instruments': 'Musical_Instruments',
    'Office': 'Office_Products',
    'Garden': 'Patio_Lawn_and_Garden',
    'Pantry': 'Prime_Pantry',
    'Pet': 'Pet_Supplies',
    'Software': 'Software',
    'Sports': 'Sports_and_Outdoors',
    'Tools': 'Tools_and_Home_Improvement',
    'Toys': 'Toys_and_Games',
    'Games': 'Video_Games',
}


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f'{func.__name__:<30} {time.perf_counter() - start:>6.2f} sec', flush=True)
        return res
    return wrapper


def load_plm(model_name='bert-base-uncased'):
    if model_name == 'all-MiniLM-L6-v2':
        model = SentenceTransformer(model_name)
        return None, model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def clean_text(text):
    # Combine operations to reduce the number of apply calls
    if isinstance(text, list):
        text = ' '.join(text)
    elif isinstance(text, dict):
        text = str(text)

    text = unidecode(html.unescape(normalize('NFKD', text)))
    text = BeautifulSoup(text, "html.parser").get_text()  # Removes all HTML tags
    text = emoji_pattern.sub('', text)
    text = re.sub(r'["\n\r]*', '', text)
    text = unprintable_pattern.sub('', text)
    text = re.sub('[\s_]+', ' ', text)
    return text


def core_n(reviews, n=5, columns=('asin', 'user_id')):
    '''repeatedly
    remove all items that have less than n reviews,
    remove all users that have less than n reviews
    '''
    while True:
        shape = reviews.shape
        for c in columns:
            vc = reviews[c].value_counts()
            reviews = reviews[reviews[c].isin(vc[vc >= n].index)]
        if reviews.shape == shape:
            return reviews


def sort_process_unsort(func):
    '''Decorator to sort the input by length, process it, and then unsort the output'''

    @wraps(func)
    def wrapper(sentences, *args, **kwargs):

        # Sort sentences by length and remember original indices
        sorted_indices, sorted_sentences = zip(*sorted(enumerate(sentences), key=lambda x: -len(x[1])))
        processed_values = func(sorted_sentences, *args, **kwargs)

        if isinstance(processed_values, np.ndarray):
            unsorted_values = np.empty_like(processed_values)
        elif isinstance(processed_values, torch.Tensor):
            unsorted_values = torch.empty_like(processed_values)
        else:
            unsorted_values = [None] * len(sentences)

        # efficiently unsort
        if isinstance(unsorted_values, (np.ndarray, torch.Tensor)):
            unsorted_values[list(sorted_indices)] = processed_values
        else:
            for original_index, value in zip(sorted_indices, processed_values):
                unsorted_values[original_index] = value

        return unsorted_values

    return wrapper


@sort_process_unsort
def bert_embedding(sentences, args, model, tokenizer):
    embeddings = []
    for batch in tqdm(
        chunked(sentences, args.batch_size),
        total=len(sentences) // args.batch_size + 1,
        desc='Embedding',
        dynamic_ncols=True,
        disable=args.quiet,
    ):

        encoded = tokenizer(
            batch,
            padding=True,
            max_length=args.max_length,
            truncation=True,
            return_tensors='pt',
        ).to(args.device)
        outputs = model(**encoded)
        cls_output = outputs.last_hidden_state[:,0,].detach().cpu()
        embeddings.append(cls_output)
    return np.concatenate(embeddings)


def sbert_embedding(sentences, args, model: SentenceTransformer):
    return model.encode(sentences, show_progress_bar=not args.quiet, batch_size=args.batch_size)


def llama_embedding(sentences, args, model, tokenizer):
    ...


def angle_embedding(sentences, args, model, tokenizer):
    embeddings = []
    for batch in tqdm(
        chunked(sentences, args.batch_size),
        total=len(sentences) // args.batch_size + 1,
        desc='Embedding',
        dynamic_ncols=True,
        disable=args.quiet,
    ):
        encoded = tokenizer(
            batch,
            padding=True,
            max_length=args.max_length,
            truncation=True,
            return_tensors='pt',
        ).to(args.device)
        embeddings.append(model(**encoded).pooler_output.detach().cpu().numpy())
    return np.concatenate(embeddings)
