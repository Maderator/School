import sys
from matplotlib import legend
import numpy as np
import math
import matplotlib.pyplot as plt

from numpy.core.numeric import cross

import entropy as ent

test_data_size = 20000
heldout_data_size = 40000

def prepare_dataset(words):
    '''
    divide dataset to train, heldout, and test parts
    '''
    td = test_data_size
    hd = heldout_data_size
    test_data = words[-td:]
    heldout_data = words[-td-hd:-td]
    train_data = words[:-td-hd]
    return train_data, test_data, heldout_data

def compute_ngram_count(words, n):
    '''
    Create dictionary of ngrams with given n with their counts in the given sequence of words
    '''
    ngram_counts = {}
    for i in range(len(words)-n+1):
        ngram = tuple(words[i:i+n]) 
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1
    return ngram_counts

def compute_ngram_counts(words, n):
    '''
    create list of dictionaries of unigrams, bigrams,..., ngrams with their counts in the given sequence of words
    n - highest number n in n-gram that we need to compute
    returns ngrams words counts in list where on position i is the (i+1)-gram
    '''
    ngram_counts = []
    for i in range(1,n+1):
        counting_words = words
        if i != n:
            counting_words = words[:-n+i]  # we dont use last n-i words for counting the (n-i)-gram occurences 
                                           # as the probabilities would not be correct as is said on slide 73 of lecture presentation
        ngram_counts.append(compute_ngram_count(counting_words, i)) 
    return ngram_counts

def extract_word_counts(train_words, n=3):
    '''
    return number of words in list of words, number of different words, number of occurences of words, and ngram counts
    n - highes number n in n-gram that we need to compute
    '''
    T = len(train_words)                     # Text size
    V = len(ent.find_all_words(train_words)) # Vocabulary size
    words_occurs = ent.compute_words_occurence(train_words)
    ngram_counts = compute_ngram_counts(train_words, n)
    return T, V, words_occurs, ngram_counts

def compute_zerogram_language_model_distribution(words, V, n=3):
    '''
    n - maximum ngram size which is used in the language modelling
    THIS FUNCTION IS NOT USED as the 1/V is constant and can be computed without dependence on any variable
    '''
    dist = {}
    for word in words[:-n+1]:
        dist[word] = 1/V
    return dist

def compute_unigram_language_model_distribution(words, unigram_counts, T, n=3):
    '''
    n - maximum ngram size which is used in the language modelling
    Models probability distribution of unigrams in words sequence 
    '''
    dist = {}
    for word in words[:-n+1]:
        dist[word] = unigram_counts[tuple([word])] / T
    return dist

def compute_bigram_language_model_distribution(words, bigram_counts, unigram_counts, n=3):
    '''
    n - maximum ngram size which is used in the language modelling
    Models probability distribution of bigrams in words sequence 
    '''
    dist = {}
    for i, word in enumerate(words[:-n+1]):
        dist[tuple([words[i], words[i+1]])] = bigram_counts[tuple([words[i], words[i+1]])] / unigram_counts[tuple([word])]
    return dist

def compute_trigram_language_model_distribution(words, trigram_counts, bigram_counts, n=3):
    '''
    n - maximum ngram size which is used in the language modelling
    Models probability distribution of trigrams in words sequence 
    '''
    dist = {}
    for i, word in enumerate(words[:-n+1]):
        dist[tuple([words[i], words[i+1], words[i+2]])] = trigram_counts[tuple([words[i], words[i+1], words[i+2]])] / bigram_counts[tuple([words[i], words[i+1]])]
    return dist

def compute_language_model_with_trigrams(words):
    '''
    returns probability distribution of language model, count of ngrams, words occurences, vocabulary size, and number of words in given words sequence
    '''
    T,V,words_occurs, ngrams = extract_word_counts(words)
    #p_zero = compute_zerogram_language_model_distribution(words, V)
    p_one = compute_unigram_language_model_distribution(words, ngrams[0],T)
    p_two = compute_bigram_language_model_distribution(words, ngrams[1], ngrams[0])
    p_three = compute_trigram_language_model_distribution(words, ngrams[2], ngrams[1])
    #p_zeor = list(p_three.items())
    #p_zeor = sorted(p_zeor, key=lambda x: x[1], reverse=True)
    #print(p_zeor[:10])
    return V, p_one, p_two, p_three, T, ngrams

def get_unigram_prob(word, unigram_dist, uniform_prob):
    '''
    return probability of word given the unigram distribution. If the word was not in the training data, then a uniform probability is returned
    '''
    if word in unigram_dist:
        return unigram_dist[word]
    else:
        return uniform_prob  # last hint on slide 93 to use uniform probability 1/|V| whenever the count_n-1(h)=0

def get_bigram_prob(words, bigram_dist, unigram_count, uniform_prob):
    '''
    return probability of bigram given the bigram distribution. If the bigram was not in the training data, then if even the unigram
    was not in the training data, uniform probability is returned, else return 0
    '''
    if tuple(words) in bigram_dist:
        return bigram_dist[tuple(words)]
    else:
        if words[0] in unigram_count:
            return 0
        else:
            return uniform_prob

def get_trigram_prob(words, trigram_dist, bigram_count, unigram_count, uniform_prob):
    '''
    return probability of trigram given the trigram distribution. If the trigram was not in the training data, then if even the bigram
    was not in the training data, uniform probability is returned, else return 0
    '''
    if tuple(words) in trigram_dist:
        return trigram_dist[tuple(words)]
    else:
        if tuple(words[0:2]) in bigram_count:
            return 0
        else:
            #if words[2] in unigram_count:
            #    return 0
            #else:
            #    return uniform_prob
            return uniform_prob

def compute_compounded_prob(trigram, lambdas, probs, ngram_counts):
    '''
    Computes the conditional probability. It is a sum of lambdas times cond. probabilities
    of given trigram, bigram, unigram and uniform part which is lambda0 / |V|
    '''
    uniform = lambdas[0]*probs[0]  # in slides it is lambda/|V| but here we have lambda * (1/|V|)
    unig = lambdas[1] * get_unigram_prob(trigram[2], probs[1], probs[0])
    big = lambdas[2] * get_bigram_prob(trigram[1:3], probs[2], ngram_counts[0], probs[0])
    trig = lambdas[3] * get_trigram_prob(trigram, probs[3], ngram_counts[1], ngram_counts[0], probs[0])
    prob = trig + big + unig + uniform
    return prob

def compute_count_unig(word, lam, prob, uniform_prob, compounded_prob):
    '''
    computes expected count for given unigram (word)
    '''
    if word in prob:
        return lam * prob[word] / compounded_prob
    else:
        #return uniform_prob
        return lam * uniform_prob / compounded_prob

def compute_count_bi(bigram, lam, bi_prob, unig_count, unif_prob, comp_prob):
    '''
    computes expected count for given bigram 
    '''
    if tuple(bigram) in bi_prob:
        return lam * bi_prob[tuple(bigram)] / comp_prob
    else:
        #if bigram[0] in unig_count:
        #    return 0
        #else:
        #    return lam * unif_prob / comp_prob
        return lam * unif_prob / comp_prob

def compute_count_tri(trigram, lam, tri_prob, bi_count, unif_prob, comp_prob):
    '''
    computes expected count for given trigram 
    '''
    if tuple(trigram) in tri_prob:
        return lam * tri_prob[tuple(trigram)] / comp_prob
    else:
        #if tuple(trigram[0:2]) in bi_count:
        #    return 0
        #else:
        #    return lam * unif_prob / comp_prob
        return lam * unif_prob / comp_prob

def compute_expected_counts(data, lambdas, probs, ngram_counts):
    '''
    computes expected counts for given data (expected counts of unigrams, bigrams, and trigrams) 
    '''
    counts = np.zeros(4)
    for i in range(2, len(data)):
        compounded_prob = compute_compounded_prob(data[i-2:i+1], lambdas, probs, ngram_counts)
        counts[0] += (lambdas[0]*probs[0]) / compounded_prob
        counts[1] += compute_count_unig(data[i], lambdas[1], probs[1], probs[0], compounded_prob)
        counts[2] += compute_count_bi(data[i-1:i+1], lambdas[2], probs[2], ngram_counts[0], probs[0], compounded_prob)
        counts[3] += compute_count_tri(data[i-2:i+1], lambdas[3], probs[3], ngram_counts[1], probs[0], compounded_prob)
    return counts

def next_lambdas(counts):
    '''
    Return new lambdas in EM algorithm given the expected counts
    '''
    c_sum = np.sum(counts)
    lambdas = np.zeros(4)
    for i in range(len(counts)):
        lambdas[i] = counts[i] / c_sum
    #print(counts)
    #print(lambdas)
    #print()
    return lambdas

def em_algorithm(data, probs, ngram_counts):
    '''
    Implementation of EM algorithm that returns lambda trained on the given words sequence (in best case on heldout data)
    '''
    lambdas = np.ones(4) * 0.25 # this could be any value > 0
    difference = 1.0
    while difference > 0.00001:
        counts = compute_expected_counts(data, lambdas, probs, ngram_counts)
        new_lambdas = next_lambdas(counts)
        difference = np.max(np.abs(lambdas - new_lambdas))
        lambdas = new_lambdas
        
    return lambdas

def compute_cross_entropy(data, lambdas, probs, ngram_counts):
    '''
    Compute cross entropy of given words sequence with trained lambdas, probability distributions, and ngram counts
    '''
    T_prime = len(data)
    cross_ent = 0
    for i in range(2, len(data)):
        cross_ent += math.log2(compute_compounded_prob(data[i-2:i+1], lambdas, probs, ngram_counts))
    return -1/T_prime * cross_ent

def test_cross_entropy_dependence_on_lambdas(data, lambdas, probs, ngram_counts):
    '''
    Run test of cross entropy value given the changes in lambdas values based on given percentages in assignemnts text.
    '''
    add_tri_lambda = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    reduce_tri_lambda = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    addition = []
    reduction = []

    new_lambdas = lambdas.copy()
    for add in add_tri_lambda:
        new_lambdas[3] += add * (1-new_lambdas[3])
        other_l_sum = np.sum(new_lambdas[0:3])
        remainder = 1-new_lambdas[3]
        new_percent = 1-(other_l_sum - remainder)/other_l_sum
        for i in range(3):
            new_lambdas[i] *= new_percent
        #print(np.sum(new_lambdas))
        addition.append((add, compute_cross_entropy(data, new_lambdas, probs, ngram_counts)))
        #addition.append(compute_cross_entropy(data, new_lambdas, probs, ngram_counts))
        new_lambdas = lambdas.copy()

    new_lambdas = lambdas.copy()
    for red in reduce_tri_lambda:
        new_lambdas[3] *= red
        other_l_sum = np.sum(new_lambdas[0:3])
        remainder = 1-new_lambdas[3]
        new_percent = 1-(other_l_sum - remainder)/other_l_sum
        for i in range(3):
            new_lambdas[i] *= new_percent
        #print(np.sum(new_lambdas))
        reduction.append((red, compute_cross_entropy(data, new_lambdas, probs, ngram_counts)))
        #reduction.append(compute_cross_entropy(data, new_lambdas, probs, ngram_counts))
        new_lambdas = lambdas.copy()

    return addition, reduction

def compute_coverage_graph(test_data, train_data):
    '''
    Computes coverage graph which is percentage of test sequence words that were seen in the train sequence words 
    '''
    word_occurs = ent.compute_words_occurence(train_data)
    count = 0
    for word in test_data:
        if word in word_occurs:
            count += 1
    return count / len(test_data)

def do_language_modelling_and_compute_cross_entropy(file, use_heldout=True):
    '''
    "Main" method of this scripy that loads words from file, prepare dataset, compute language model and tests the cross entropy
    based on the changed lambdas.
    '''
    words = ent.load_words_from_file(file)
    train, test, heldout = prepare_dataset(words)
    print("coverage graph = {}".format(compute_coverage_graph(test, train)))
    V, p_one, p_two, p_three, train_data_size, ngram_counts = compute_language_model_with_trigrams(train)
    V = train_data_size # Hint on slide 93 says that the approximated size of "vocabulary" V should be equal to number of all words from training data
    p_zero = 1/V
    probs = [p_zero, p_one, p_two, p_three]

    # compute lambdas
    if use_heldout == True:
        lambdas = em_algorithm(heldout, probs, ngram_counts)
    else:
        lambdas = em_algorithm(train, probs, ngram_counts)
    print("lambdas=", lambdas)
    print(np.sum(lambdas))

    # compute cross entropy of test data with use of probabilities trained on train data and lambdas computed on heldout data
    #cross_entropy = compute_cross_entropy(heldout, lambdas, probs, ngram_counts)
    #cross_entropies = test_cross_entropy_dependence_on_lambdas(heldout, lambdas, probs, ngram_counts)
    words = ent.load_words_from_file("TEXTEN1.txt")
    _, test, _ = prepare_dataset(words)
    print("coverage graph = {}".format(compute_coverage_graph(test, train)))

    cross_entropy = compute_cross_entropy(test, lambdas, probs, ngram_counts)
    addition, reduction = test_cross_entropy_dependence_on_lambdas(test, lambdas, probs, ngram_counts)

    #for ce in addition:
    #    print(ce)
    #print()
    #for ce in reduction:
    #    print(ce)
    #print()

    return cross_entropy, addition, reduction

def plot_experiment(data, legend_name=''):
    '''
    Plot the data
    '''
    x = [i for i in range(len(data))]
    plt.plot(x, data, label=legend_name)

cross_entropy, addition, reduction = do_language_modelling_and_compute_cross_entropy("TEXTCZ1.txt", use_heldout=True)
#add_dif = np.array(addition)-cross_entropy
#plot_experiment(add_dif, legend_name="Czech lang. addition to trigram smoothing parameter")
#red_dif = np.array(reduction) - cross_entropy
#plot_experiment(red_dif, legend_name="Czech lang. reduction to trigram smoothing parameter")

print(cross_entropy)

cross_entropy, addition, reduction = do_language_modelling_and_compute_cross_entropy("TEXTEN1.txt", use_heldout=True)
#add_dif = np.array(addition)-cross_entropy
#plot_experiment(add_dif, legend_name="English lang. addition to trigram smoothing parameter")
#red_dif = np.array(reduction) - cross_entropy
#plot_experiment(red_dif, legend_name="English lang. reduction to trigram smoothing parameter")

print(cross_entropy)

#plt.title("Cross entropy based on change of smoothing parameters")
#plt.xlabel("number of test case")
#plt.ylabel("Cross entropy difference from the default smoothing parameters")
#plt.legend(loc="upper left")
##plt.yscale('log')
##plt.loglog()
#plt.show()