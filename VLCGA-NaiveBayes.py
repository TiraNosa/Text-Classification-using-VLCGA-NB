from NaiveBayesClassifier import NaiveBayesClassifier
import GeneticAlgorithm
import time

naiveBayes = NaiveBayesClassifier()

# read training data
df = naiveBayes.loadCSV('V1.4_Training_new.csv')
contractions = naiveBayes.loadCSV('Contractions.csv', 'contraction', 'text1').to_dict('split')

df['label'] = df.sentence.str.extract(r'((\b\w+)[\.?!\s]*$)')[0]

df['label'] = df.label.map(lambda x: 'suggestion' if x == '1' else 'nonsuggestion')

# preprocessing training data
df = naiveBayes.Preprocessing(df, contractions)

# First element is an array containing the words and second element is the type of sentiment.
# We get rid of the words smaller than 2 characters and we use lowercase for everything.
allsentences = [(row['sentence'], row['label']) for index, row in df.iterrows()]
trainingsentences = [(([e.lower() for e in words.split() if len(e) >= 3], sentiment)) for (words, sentiment) in allsentences]

GA = GeneticAlgorithm
chromosomeManager = GA.ChromosomeManager()
chromosomeManager.sentences = trainingsentences
word_features = naiveBayes.getWordFeatures(trainingsentences)
for feature in word_features:
    chromosomeManager.addFeatures(feature)

pop = GA.Population(chromosomeManager, 100, True)
ga = GA.GA(chromosomeManager)
then = time.time()
for i in range(0, 50):
    print("Generation: ", i)
    pop = ga.evolvePopulation(pop)
    now = time.time()
    diff = now - then
    f = open('VLGCA-SelectedSuggestionWords-TaskB-2.csv', 'a')
    f.write(str(i + 1) + ',' + str(pop.getFittest().chromosomeSize()) + ',' + str(
        pop.getFittest().getAccuracyRate().get('accuracy'))
            + ',' + str(pop.getFittest().getAccuracyRate().get('suggestion precision'))
            + ',' + str(pop.getFittest().getAccuracyRate().get('suggestion recall'))
            + ',' + str(pop.getFittest().getAccuracyRate().get('nonsuggestion precision'))
            + ',' + str(pop.getFittest().getAccuracyRate().get('nonsuggestion recall'))
            + ',' + str(pop.getFittest().getFMeasureSuggestion())
            + ',' + str(pop.getFittest().getFMeasureNonSuggestion())
            + ',' + str(pop.getFittest().getFitness()) + ',' + str(pop.getFittest())
            + ',' + str(diff) + '\n')
    f.close()

    d = open('last population-SelectedSuggestionWords-TaskB-2.txt', 'w')
    for j in range(0, pop.populationSize()):
        d.write(str(pop.getChromosome(j)) + '\n')
    d.close()