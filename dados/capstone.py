import string
import unidecode
import re
import unidecode
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def preprocess_text(raw_corpus):
    raw_corpus = raw_corpus.str.replace('[\r\n]',' ')
    pt_stopwords = set([unidecode.unidecode(i) for i in nltk.corpus.stopwords.words('portuguese')])    
    def preprocess(x, *stopwords_list):
        x = unidecode.unidecode(re.sub(r'(?:@[\w_]+)', '', x) # eliminando os valores com @.. (citacoes)
        stemmer = nltk.stem.RSLPStemmer()
        x = unidecode.unidecode(re.sub('['+string.punctuation+'«»'+']', ' ', x.lower()))
        x = [stemmer.stem(w) for w in x.split() if w not in set(pt_stopwords)]
        return ' '.join(x)

    clean_text = raw_corpus.apply(preprocess, args = pt_stopwords)
    clean_text = clean_text.str.replace('\d+', '')
    return clean_text

def preprocess_text_fasttext(raw_corpus):
    raw_corpus = raw_corpus.str.replace('[\r\n]',' ')  
    def preprocess(x):
        x = re.sub('['+string.punctuation+'«»'+']', ' ', x.lower())
        x = re.sub('[ ]{2,}', ' ', x.lower()).strip()
        return x

    clean_text = raw_corpus.apply(preprocess)
    clean_text = clean_text.str.replace('\d+', '')
    return clean_text

def get_word_freq_by_class(data, classe):
    class_news = data.clean_text.loc[data.classe == classe].values.astype('U')
    count_vectorizer = CountVectorizer().fit(class_news)
    class_bow = count_vectorizer.transform(class_news)
    class_vocab = list(count_vectorizer.get_feature_names())
    counts = class_bow.sum(axis=0).A1
    class_word_counts = pd.DataFrame(list(dict(zip(class_vocab, counts)).items()), 
                                   columns = ['word','freq']).sort_values('freq', ascending = False)
    return class_word_counts