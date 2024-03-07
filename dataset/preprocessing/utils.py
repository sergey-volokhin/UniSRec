import html
import os
import re
import string
import time
from functools import wraps
from unicodedata import normalize

from bs4 import BeautifulSoup
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from unidecode import unidecode

unprintable_pattern = re.compile(f'[^{re.escape(string.printable)}]')
emoji_pattern = re.compile(
    u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF\U00002500-\U00002BEF\U00002702-\U000027B0"
    u"\U00002702-\U000027B0\U000024C2-\U0001F251\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff\u2640-\u2642\u2600-\u2B55\u200d\u23cf\u23e9\u231a\ufe0f\u3030]+",
    flags=re.UNICODE,
)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(f'{func.__name__:<30} {time.perf_counter() - start:>6.2f} sec')
        return res
    return wrapper


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_plm(model_name='bert-base-uncased'):
    if model_name == 'all-MiniLM-L6-v2':
        model = SentenceTransformer('all-MiniLM-L6-v2')
        tokenizer = None
        return tokenizer, model
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
