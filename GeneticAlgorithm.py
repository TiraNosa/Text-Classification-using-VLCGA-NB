import random
from random import randint
from NaiveBayesClassifier import NaiveBayesClassifier
import time


class Feature:
    def __init__(self, feature):
        self.feature = None
        if feature is not None:
            self.feature = feature

    def getFeature(self):
        return self.feature

    def __repr__(self):
        return str(self.feature + " ")



class Population:
    def __init__(self, chromosomeManager, populationSize, initialize):
        self.chromosomes = []
        for i in range(0, populationSize):
            self.chromosomes.append(None)

        if initialize:
            for i in range(0, populationSize):
                numberOfFeatures = randint(1, chromosomeManager.numberOfFeatures())
                newChromosome = Chromosome(chromosomeManager, numberOfFeatures)
                newChromosome.generateIndividual()
                self.saveChromosome(i, newChromosome)

    def __setitem__(self, key, value):
        self.chromosomes[key] = value

    def __getitem__(self, index):
        return self.chromosomes[index]

    def saveChromosome(self, index, chromosome):
        self.chromosomes[index] = chromosome

    def getChromosome(self, index):
        return self.chromosomes[index]

    def getFittest(self):
        fittest = self.chromosomes[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness() <= self.getChromosome(i).getFitness():
                fittest = self.getChromosome(i)
        return fittest

    def populationSize(self):
        return len(self.chromosomes)


class Chromosome:
    def __init__(self, chromosomeManager, numSelectedFeature, chromosome=None):
        self.chromosomeManager = chromosomeManager
        self.chromosome = []
        self.fitness = 0.0
        self.accuracy = 0.0
        self.reduction = 0.0
        self.suggestion = 0.0
        self.nonsuggestion = 0.0

        if chromosome is not None:
            self.chromosome = chromosome
        else:
            for i in range(0, numSelectedFeature):
                self.chromosome.append(None)

    def __len__(self):
        return len(self.chromosome)

    def __getitem__(self, index):
        return self.chromosome[index]

    def __setitem__(self, key, value):
        self.chromosome[key] = value

    def __repr__(self):
        geneString = "|"
        for i in range(0, self.chromosomeSize()):
            geneString += str(self.getFeature(i)) + "|"
        return geneString

    def generateIndividual(self):
        for featureIndex in range(0, self.chromosomeSize()):
            while True:
                selectedFeature = randint(0, self.chromosomeManager.numberOfFeatures() - 1)
                if not self.containsFeature(self.chromosomeManager.getFeature(selectedFeature)):
                    break

            self.setFeature(featureIndex, self.chromosomeManager.getFeature(selectedFeature))

    def setFeature(self, chromosomePosition, feature):
        self.chromosome[chromosomePosition] = feature
        self.fitness = 0.0

    def getFeature(self, chromosomePosition):
        return self.chromosome[chromosomePosition]

    def getFitness(self):
        if self.fitness == 0:
            self.getAccuracyRate()
            self.fitness = (
                self.accuracy.get('accuracy') + self.getFMeasureSuggestion() + self.getFMeasureNonSuggestion())
        return self.fitness

    def getAccuracyRate(self):
        if self.accuracy == 0:
            naiveBayes = NaiveBayesClassifier()
            naiveBayes.word_features = self.chromosome
            classifier = naiveBayes.NaiveBayes(self.chromosomeManager.sentences)
            self.accuracy = naiveBayes.getAccuracy(classifier, self.chromosomeManager.sentences)
        return self.accuracy

    def getFMeasureSuggestion(self):
        if self.suggestion == 0:
            self.suggestion = 0.0 if self.accuracy.get('suggestion recall') == 0 and self.accuracy.get(
                'suggestion precision') == 0 else 2 * (
                (self.accuracy.get('suggestion recall') * self.accuracy.get('suggestion precision')) / (
                    self.accuracy.get('suggestion recall') +
                    self.accuracy.get('suggestion precision')))
        return self.suggestion

    def getFMeasureNonSuggestion(self):
        if self.nonsuggestion == 0:
            self.nonsuggestion = 0.0 if self.accuracy.get('nonsuggestion recall') == 0 and self.accuracy.get(
                'nonsuggestion precision') == 0 else 2 * (
                    (self.accuracy.get('nonsuggestion recall') * self.accuracy.get('nonsuggestion precision')) / (
                    self.accuracy.get('nonsuggestion recall') +
                    self.accuracy.get('nonsuggestion precision')))
        return self.nonsuggestion

    def chromosomeSize(self):
        return len(self.chromosome)

    def containsFeature(self, feature):
        return feature in self.chromosome


class ChromosomeManager:
    selectedFeatures = []
    sentences = []

    def addFeatures(self, feature):
        self.selectedFeatures.append(feature)

    def getFeature(self, index):
        return self.selectedFeatures[index]

    def numberOfFeatures(self):
        return len(self.selectedFeatures)

    def setSentences(self, sentences):
        self.sentences = sentences


class GA:
    def __init__(self, chromosomeManager):
        self.chromosomeManager = chromosomeManager
        self.mutationRate = 0.3
        self.elitism = False
        self.crossoverRate = 0.7
        self.tournamentSize = 5

    def evolvePopulation(self, pop):
        tempPopulation = Population(self.chromosomeManager,
                                    pop.populationSize() + int(pop.populationSize() * self.crossoverRate) + int(
                                        pop.populationSize() * self.mutationRate), False)
        newPopulation = Population(self.chromosomeManager, pop.populationSize(), False)
        then = time.time()
        for i in range(0, pop.populationSize()):
            tempPopulation.saveChromosome(i, pop.getChromosome(i))

        now = time.time()
        diff = now - then
        print("Finish adding the parent chromosome ", diff)
        idx = pop.populationSize()
        then = time.time()
        while idx < pop.populationSize() + int(pop.populationSize() * self.crossoverRate):
            randomId = randint(0, pop.populationSize() - 1)
            parent1 = pop.getChromosome(randomId)
            while True:
                randomId = randint(0, pop.populationSize() - 1)
                parent2 = pop.getChromosome(randomId)
                if not parent1 == parent2:
                    break

            child = self.crossover(parent1, parent2)
            tempPopulation.saveChromosome(idx, child)
            idx += 1

            # child = self.crossover2(parent1, parent2)
            # tempPopulation.saveChromosome(idx, child)
            # idx += 1

            child = self.crossover3(parent1, parent2)
            if child.chromosomeSize() == 0:
                while True:
                    randomId = randint(0, pop.populationSize() - 1)
                    parent1 = pop.getChromosome(randomId)
                    while True:
                        randomId = randint(0, pop.populationSize() - 1)
                        parent2 = pop.getChromosome(randomId)
                        if not parent1 == parent2:
                            break
                    child = self.crossover3(parent1, parent2)
                    if child.chromosomeSize() > 0:
                        break
            tempPopulation.saveChromosome(idx, child)
            idx += 1

        now = time.time()
        diff = now - then
        print("Finish crossover", diff)
        then = time.time()
        for i in range(pop.populationSize() + int(pop.populationSize() * self.crossoverRate),
                       tempPopulation.populationSize()):
            randomId = randint(0, pop.populationSize() - 1)
            parent = pop.getChromosome(randomId)
            child = self.mutate(parent)
            tempPopulation.saveChromosome(i, child)
        now = time.time()
        diff = now - then
        print("Finish mutation", diff)
        then = time.time()
        for i in range(0, tempPopulation.populationSize()):
            min_idx = i
            for j in range(i + 1, tempPopulation.populationSize()):
                if tempPopulation.getChromosome(min_idx).getFitness() < tempPopulation.getChromosome(j).getFitness():
                    min_idx = j

            temp = tempPopulation.getChromosome(i)
            tempPopulation.saveChromosome(i, tempPopulation.getChromosome(min_idx))
            tempPopulation.saveChromosome(min_idx, temp)
        now = time.time()
        diff = now - then
        print("Finish sorting", diff)
        then = time.time()
        for i in range(0, pop.populationSize()):
            newPopulation.saveChromosome(i, tempPopulation.getChromosome(i))
        now = time.time()
        diff = now - then
        print("Finish new chromosome", diff)
        return newPopulation

    def crossover(self, parent1, parent2):
        features = []
        for i in range(0, parent1.chromosomeSize()):
            features.append(parent1.getFeature(i))

        for i in range(0, parent2.chromosomeSize()):
            if parent2.getFeature(i) not in features:
                features.append(parent2.getFeature(i))

        child = Chromosome(self.chromosomeManager, len(features))

        for i in range(0, len(features)):
            child.setFeature(i, features[i])

        return child

    def crossover2(self, parent1, parent2):
        if (len(parent1) < len(parent2)):
            selectedParent1 = parent1
            selectedParent2 = parent2
        else:
            selectedParent1 = parent2
            selectedParent2 = parent1

        child = Chromosome(self.chromosomeManager, len(selectedParent1))
        startPos = randint(0, selectedParent1.chromosomeSize() - 1)
        endPos = randint(0, selectedParent1.chromosomeSize() - 1)

        for i in range(0, child.chromosomeSize()):
            if startPos < endPos and i > startPos and i < endPos:
                child.setFeature(i, selectedParent1.getFeature(i))
            elif startPos > endPos:
                if not (i < startPos and i > endPos):
                    child.setFeature(i, selectedParent1.getFeature(i))

        for i in range(0, selectedParent2.chromosomeSize()):
            if not child.containsFeature(selectedParent2.getFeature(i)):
                for ii in range(0, child.chromosomeSize()):
                    if child.getFeature(ii) == None:
                        child.setFeature(ii, selectedParent2.getFeature(i))
                        break

        return child

    def crossover3(self, parent1, parent2):
        features = []

        for i in range(0, parent1.chromosomeSize()):
            if parent2.containsFeature(parent1.getFeature(i)):
                features.append(parent1.getFeature(i))

        child = Chromosome(self.chromosomeManager, len(features))
        for i in range(0, len(features)):
            child.setFeature(i, features[i])

        return child

    def mutate(self, chromosome):
        child = Chromosome(self.chromosomeManager, chromosome.chromosomeSize())
        #copy the chromosome
        for i in range(0, len(chromosome)):
            child.setFeature(i, chromosome.getFeature(i))

        idx = -1
        #max feature will be changed is the diff between the total of number of features and the chromosome size
        maxFeatureChanged = self.chromosomeManager.numberOfFeatures() - chromosome.chromosomeSize()

        for chromosomePos1 in range(0, chromosome.chromosomeSize()):
            #if the max feature is zero than break the process
            if maxFeatureChanged == 0:
                break
            #change the gene if the random number is less than the mutation rate
            if random.random() < self.mutationRate:
                while True:
                    idx += 1
                    #if the child doesn't contain the feature, change the feature
                    if not child.containsFeature(self.chromosomeManager.getFeature(idx)):
                        feature1 = self.chromosomeManager.getFeature(idx)
                        child.setFeature(chromosomePos1, feature1)
                        maxFeatureChanged -= 1
                        break
        return child

    def tournamentSelection(self, pop):
        tournament = Population(self.chromosomeManager, self.tournamentSize, False)
        for i in range(0, self.tournamentSize):
            randomId = randint(0, pop.populationSize() - 1)
            tournament.saveChromosome(i, pop.getChromosome(randomId))
        fittest = tournament.getFittest()
        return fittest