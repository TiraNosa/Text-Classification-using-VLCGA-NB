import nltk
import pandas as pd
import nltk.classify.util, nltk.metrics
from nltk import collections
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import string


class NaiveBayesClassifier:
    def __init__(self):
        self.word_features = None

    def loadCSV(self, filename, column1=None, column2=None):
        if column1 and column2 is not None:
            df = pd.read_csv(filename, usecols=[column1, column2])
        else:
            df = pd.read_csv(filename, sep='delimiter', header=None)
            mapping = {0: 'sentence'}
            df.index.names = list(map(lambda name: mapping.get(name, name), df.index.names))
            df.rename(columns=mapping, inplace=True)
            # df.rename(columns={'id,sentence,label': 'sentence'}, inplace=True)
        return df

    def Preprocessing(self, df, contractions):
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
            row[0] = ' '.join([str(x) for x in [contractionsDict[t] if t in contractionsDict.keys() else t for t in
                                                [e.lower() for e in row[0].split()]]])

        # remove stopword
        stop_words = []
        for word in stopwords.words('english'):
            stop_words.append(word) if ('not' not in word and 'no' not in word) else stop_words

        # remove punctuation
        tokenizer = RegexpTokenizer(r'\w+')
        for index, row in df.iterrows():
            word_tokens = tokenizer.tokenize(row[0])
            row[0] = ' '.join([w for w in word_tokens if not w.lower() in stop_words])

        # using lemmetizer
        wordnet_lemmatizer = WordNetLemmatizer()
        for index, row in df.iterrows():
            row[0] = ' '.join(wordnet_lemmatizer.lemmatize(t) for t in row[0].split())

        # remove non-english word
        english_words = set(nltk.corpus.words.words())
        for index, row in df.iterrows():
            word_tokens = tokenizer.tokenize(row[0])
            row[0] = " ".join(w for w in word_tokens if w.lower() in english_words or not w.isalpha())

        # remove non-alphabetic characters
        for index, row in df.iterrows():
            word_tokens = tokenizer.tokenize(row[0])
            row[0] = " ".join(w for w in word_tokens if w.isalpha())

        return df

    def load_words(self):
        with open('words_alpha.txt') as word_file:
            valid_words = set(word_file.read().split())

        return valid_words

    def get_words_in_sentences(self, sentences):
        all_words = []
        suggestionWords=[]
        nonsuggestionWords=[]
        for (words, sentiment) in sentences:
            # all_words.extend(words)
            if sentiment == 'suggestion':
                suggestionWords.extend(words)
            else:
                nonsuggestionWords.extend(words)

        for suggestion in suggestionWords:
            if suggestion not in nonsuggestionWords:
                all_words.append(suggestion)
        print(all_words)
        return all_words

    # Eliminating features with extremely low frequency
    def get_word_features(self, wordlist):
        fdist = nltk.FreqDist(wordlist)
        # word_features = dict((el, 0) for el in [x[0] for x in list(filter(lambda x: x[1] >= 6, fdist.items()))])
        return fdist

    def getWordFeatures(self, trainingsentences):
        word_features = self.get_word_features(self.get_words_in_sentences(trainingsentences))
        return word_features

    def extract_features_unigram(self, document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            if isinstance(word, str):
                features['contains(%s)' % word] = (word in document_words)
        return features

    def NaiveBayes(self, sentences):
        training_set = nltk.classify.apply_features(self.extract_features_unigram, sentences)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        return classifier

    def getAccuracy(self, classifier, sentences):
        test_set = nltk.classify.apply_features(self.extract_features_unigram, sentences[:500])
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        classifierResult = {}

        for i, (feats, label) in enumerate(test_set):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)

        classifierResult['accuracy'] = nltk.classify.util.accuracy(classifier, test_set)

        classifierResult['suggestion precision'] = 0.0 if nltk.precision(refsets['suggestion'], testsets[
            'suggestion']) == None else nltk.precision(refsets['suggestion'], testsets['suggestion'])

        classifierResult['suggestion recall'] = 0.0 if nltk.recall(refsets['suggestion'],
                                                                   testsets['suggestion']) == None else nltk.recall(
            refsets['suggestion'], testsets['suggestion'])

        classifierResult['nonsuggestion precision'] = 0.0 if nltk.precision(refsets['nonsuggestion'], testsets[
            'nonsuggestion']) == None else nltk.precision(refsets['nonsuggestion'], testsets['nonsuggestion'])

        classifierResult['nonsuggestion recall'] = 0.0 if nltk.recall(refsets['nonsuggestion'],
                                                                      testsets[
                                                                          'nonsuggestion']) == None else nltk.recall(
            refsets['nonsuggestion'], testsets['nonsuggestion'])

        return classifierResult