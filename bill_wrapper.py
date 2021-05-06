from scipy.spatial.distance import cosine
import pickle
import build_tree
import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips
from persim import bottleneck, sliced_wasserstein
import io
import pandas as pd

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

def load_dict(dict_path="data/crosslingual/dictionaries/en-es.0-5000.txt"):
    return pd.read_csv(dict_path, names=["src", "tgt"], delim_whitespace=True)


def multi_key_dict(words, dict_):
    out = []
    for word in words:
        if word in dict_:
            out.append(dict_[word])
    return np.asarray(out)



def cosine_dist(a, b):
    sim = cosine(a,b)
    return np.arccos(1.0 - sim)  / np.pi

def main():
    embeddings = ['data/wiki.en.vec', 'data/wiki.es.vec', 'data/wiki.zh.vec', 'data/wiki.ko.vec', 'data/wiki.ru.vec', 'data/wiki.ja.vec', 'data/wiki.de.vec', 'data/wiki.nl.vec', 'data/wiki.fr.vec', 'data/wiki.ar.vec', 'data/wiki.fi.vec', 'data/wiki.hu.vec']
    dictionaries = ['en-en.0-5000.txt', 'en-es.0-5000.txt', 'en-zh.0-5000.txt', 'en-ko.0-5000.txt', 'en-ru.0-5000.txt', 'en-ja.0-5000.txt', 'en-de.0-5000.txt', 'en-nl.0-5000.txt', 'en-fr.0-5000.txt', 'en-ar.0-5000.txt', 'en-fi.0-5000.txt', 'en-hu.0-5000.txt']
    languages = ['English', 'Spanish', 'Mandarin', 'Korean', 'Russian', 'Japanese', 'German', 'Dutch', 'French', 'Arabic', 'Finnish', 'Hungarian']
    nmax = 50000  # maximum number of word embeddings to load
    data = dict()
    for l_names, path, mydpath in zip(languages, embeddings, dictionaries):
        emb, id2word, word2id = load_vec(path, nmax)
        en_to_x_dict = load_dict("data/crosslingual/dictionaries/" + mydpath)
        src = en_to_x_dict["tgt"].values
        ids = multi_key_dict(src, word2id)
        data[l_names] = emb[ids,:][:500,:]
        dgrms = dict()
    for language in languages[:-2]:
        rips = Rips(maxdim=2)
        print("Language: ", language)
        dgrms[language] = rips.fit_transform(data[language], metric=cosine_dist)

    langs = list(dgrms.keys())
    langs.append('')
    q1, tiny = build_tree.hclustering(dgrms,homology=1,dist='sw', linkage="complete")
    print("Complete linkage")
    q1.display(langs)
    build_tree.make_latex_tree(q1,langs)














