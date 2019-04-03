import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

def Preprocessing(df, contractions):
    pd.options.mode.chained_assignment = None
    contractionsDict = {}
    for i in contractions['data']:
        contractionsDict[i[0]] = i[1]

    # remove url
    df['sentence'] = df['sentence'].str.replace('http\S+|www.\S+', '', case=False)

    # remove number
    df['sentence'] = df['sentence'].str.replace('\d+', '')

    # remove hashtags
    df['sentence'] = df['sentence'].str.replace('#(\w+)', '')

    # change all text with contraction
    for index, row in df.iterrows():
        row[1] = ' '.join([str(x) for x in [contractionsDict[t] if t in contractionsDict.keys() else t for t in
                                            [e.lower() for e in row[1].split()]]])

    # remove stopword
    stop_words = []
    for word in stopwords.words('english'):
        stop_words.append(word) if ('not' not in word and 'no' not in word) else stop_words

    # remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    for index, row in df.iterrows():
        word_tokens = tokenizer.tokenize(row[1])
        row[1] = ' '.join([w for w in word_tokens if not w.lower() in stop_words])

    # using lemmetizer
    wordnet_lemmatizer = WordNetLemmatizer()
    for index, row in df.iterrows():
        row[1] = ' '.join(wordnet_lemmatizer.lemmatize(t) for t in row[1].split())

    # remove non-english word
    english_words = set(nltk.corpus.words.words())
    for index, row in df.iterrows():
        word_tokens = tokenizer.tokenize(row[1])
        row[1] = " ".join(w for w in word_tokens if w.lower() in english_words or not w.isalpha())

    # remove non-alphabetic characters
    for index, row in df.iterrows():
        word_tokens = tokenizer.tokenize(row[1])
        row[1] = " ".join(w for w in word_tokens if w.isalpha())

    return df

def loadCSV(filename, column1=None, column2=None):
    if column1 and column2 is not None:
       df = pd.read_csv(filename, usecols=[column1, column2])
    else:
        df = pd.read_csv(filename, sep='delimiter', header=None)
        mapping = {0: 'sentence'}
        df.index.names = list(map(lambda name: mapping.get(name, name), df.index.names))
        df.rename(columns=mapping, inplace=True)
        # df.rename(columns={'id,sentence,label': 'sentence'}, inplace=True)
    return df