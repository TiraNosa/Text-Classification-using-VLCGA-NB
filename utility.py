import csv

import nltk
import pandas as pd

# This will create and write into a new CSV
from nltk import RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords

# This will create and write into a new CSV
def write_csv(sent_list, label_list, out_path):
    filewriter = csv.writer(open(out_path, "w+", newline='',encoding="utf-8"))
    count = 0
    header = ['id', 'sentence', 'label']
    filewriter.writerow(i for i in header)
    for ((id, sent), label) in zip(sent_list, label_list):
        filewriter.writerow([id, sent, label])

# This reads CSV a given CSV and stores the data in a list
def read_csv(data_path):
    file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
    sent_list = []
    next(file_reader)
    for row in file_reader:
        id = row[0]
        sent = row[1]

        sent_list.append((id, sent))
    return sent_list



def loadCSV(filename, column1=None, column2=None):
    if column1 and column2 is not None:
        df = pd.read_csv(filename, usecols=[column1, column2])
    else:
        df = pd.read_csv(filename, sep='delimiter', header=0)
        mapping = {0: 'sentence'}
        df.index.names = list(map(lambda name: mapping.get(name, name), df.index.names))
        df.rename(columns=mapping, inplace=True)
        # df.rename(columns={'id,sentence,label': 'sentence'}, inplace=True)
    return df


def readFile(file):
    df = pd.read_csv(file)
    df.head()

    # add a column encoding the label as an integer
    # because categorical variables are often better represented by integers than strings
    col = ['label', 'sentence']
    df = df[col]
    df = df[pd.notnull(df['sentence'])]
    df.columns = ['label', 'sentence']
    df['category_id'] = df['label']
    df['label_str'] = df['label'].replace({1: 'suggestion', 0: 'non-suggestion'})
    df.head()
    return df

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

def get_word_features(wordlist):
    fdist = nltk.FreqDist(wordlist)
    word_features = dict((el, 0) for el in [x[0] for x in list(filter(lambda x: x[1] >= 6, fdist.items()))])
    return word_features

def getWordFeatures(trainingsentences):
    word_features = get_word_features(get_words_in_sentences(trainingsentences))
    return word_features

def get_words_in_sentences(sentences):
    all_words = []
    for (words, sentiment) in sentences:
        if sentiment == 'suggestion':
            all_words.extend(words)
    return all_words