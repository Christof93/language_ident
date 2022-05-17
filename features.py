from collections import defaultdict, abc
import numpy as np

class sparse_vector(abc.Iterable):
    """
    A simple sparse vector representation as list of tuples. 
    With a size argument for true size.
    """
    def __init__(self, index_value_tuples, size) -> None:
        self.size = size
        self._non_zero_values = index_value_tuples

    def __repr__(self) -> str:
        return str(self._non_zero_values)

    def __iter__(self):
        return iter(self._non_zero_values)

def get_n_grams(collection, n):
    """
    build n-gram tuples over a collection of elements.
    """
    collections = [collection[i:] for i in range(n)]
    ngrams = []
    for ngram in zip(*collections):
        ngrams.append(ngram)
    return ngrams

def collect_all_n_grams(all_texts):
    """
    Computes all the distinct ngrams for a collection of textual strings, and indexes them.
    """
    distinct_n_grams = set()
    for text in all_texts:
        for i in range(1,5):
            distinct_n_grams |= set(get_n_grams(clean_text(text), i))
    look_up = {ngram:i for i,ngram in enumerate(distinct_n_grams)}
    return look_up    

def clean_text(text):
    """
    filter non-alphabetic content
    """
    return [char for char in text if char.isalpha() or char == ' ']

def build_feature_vector(text, n_gram_lookup, n_gram_window = (1,5)):
    """
    Return a sparse vector with n-gram counts for a specific text
    """
    unknown_ngrams = 0
    all_ngrams_present = []
    count_n_grams = defaultdict(int)
    for i in range(*n_gram_window):
        all_ngrams_present += get_n_grams(clean_text(text), i)
    for ngram in all_ngrams_present:
        if ngram in n_gram_lookup:
            count_n_grams[n_gram_lookup[ngram]] += 1
        else:
            unknown_ngrams += 1
    if unknown_ngrams > 0:
        pass
        #print(f'warning: {unknown_ngrams} unknown ngrams detected')
    vector = sparse_vector(list(count_n_grams.items()), len(n_gram_lookup))
    return vector


if __name__ == "__main__":
    data = [
        'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus egestas augue sed diam fringilla, ac egestas dui venenatis.',
        'Quisque convallis vitae mi et suscipit. Duis aliquet neque dictum diam dictum malesuada. Maecenas fermentum urna tellus, eget',
        'lobortis lorem suscipit quis. Pellentesque efficitur mauris eu condimentum finibus. Vivamus iaculis orci at pulvinar porttitor.',
        'Morbi condimentum consectetur nisl sit amet malesuada. Donec facilisis tempor iaculis. Phasellus consectetur ut mauris nec malesuada.',
        'Vestibulum id dolor facilisis, semper nulla ac, porta nulla. Etiam vel malesuada ligula. Nam porttitor posuere elit, non efficitur ex commodo non.',
    ]
    distinct_n_grams = collect_all_n_grams(data)
    for i, (key,val) in enumerate(distinct_n_grams.items()):
        print (key,':', val)
        if i >10:
            break
    for text in data:
        vector = build_feature_vector(text, distinct_n_grams)
    print(vector.size)
    print(len(list(vector)))
    

