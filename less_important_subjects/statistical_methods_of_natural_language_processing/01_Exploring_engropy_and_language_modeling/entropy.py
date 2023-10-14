import math
import random
import numpy as np
import itertools

def load_words_from_file(file):
    with open(file, 'r', encoding='iso-8859-2') as reader:
        words = [line.strip('\n') for line in reader.readlines()] # load words from file without endline symbol
    return words

def compute_bigram_count(words):
    '''
    Count bigrams occurences
    '''
    bigram_counts = {}
    for i in range(len(words)-1):
        word1 = words[i]
        word2 = words[i+1]
        bigram = (word1, word2)
        if bigram in bigram_counts:
            bigram_counts[bigram] += 1
        else:
            bigram_counts[bigram] = 1
    return bigram_counts

def compute_words_occurence(words):
    '''
    Count words occurences
    '''
    words_occurs = {}
    for word in words:
        if word in words_occurs:
            words_occurs[word] += 1
        else:
            words_occurs[word] = 1
    return words_occurs

def compute_word_occurence_probability(words):
    '''
    Count the probability of word in given words sequence
    ''' 
    words_occurs = compute_words_occurence(words)
    words_prob = words_occurs
    total_words = len(words)
    for w, count in words_prob.items():
        words_prob[w] = count / total_words
    return words_prob

def compute_joint_prob(words):
    ''' 
    return dictionary with probability P(i,j) that at any position in the text you will find the word i followed immediately by the word j
    '''
    bigram_count = compute_bigram_count(words)
    number_of_words = len(words) # we use number of words (not bigrams) because we are computing probability for each word i, NOT for each bigram
    probability = bigram_count
    for k, v in probability.items():
        probability[k] = v/number_of_words
    return probability



def compute_conditional_prob(joint_prob, word_prob):
    '''
    joint_prob - precomputed joint probability 
    word_prob  - precomputed probability of word occurence in text
    return dictionary with probability P(j|i) that if word i occurs in the text then word j will follow
    '''
    cond_prob = joint_prob.copy()
    for bi, joint_p in cond_prob.items():
        cond_prob[bi] = joint_p / word_prob[bi[0]]
    return cond_prob
    

def compute_conditional_entropy_of_words_list(words):
    '''
    Computes conditional entropy of given words sequence by first computing joint and conditional probability.
    '''
    joint_prob = compute_joint_prob(words)
    word_prob = compute_word_occurence_probability(words)
    conditional_prob = compute_conditional_prob(joint_prob, word_prob)

    #print(word_prob)
    entropy = 0
    for bigram, cond_prob in conditional_prob.items():
        #jp = joint_prob[bigram]
        #print(str(bigram) + " " + str(cond_prob) + " " + str(jp))
        entropy += joint_prob[bigram] * math.log(cond_prob, 2)
    return -entropy


def compute_conditional_entropy_of_file(file):
    '''
    entropy H(J|I) = - \sum_{i \in I, j \in J}  P(i,j) * log_2 P(j|i)
    '''
    words = load_words_from_file(file)
    return compute_conditional_entropy_of_words_list(words)

def compute_perplexity_from_entropy(entropy):
    '''
    perplexity PX(P(J|I)) = 2^{H(J|I)}
    '''
    return 2 ** entropy

def find_all_characters(words):
    '''
    Finds set of all characters in the given text (words sequence)
    '''
    chars = set()
    for word in words:
        for ch in word:
            chars.add(ch)
    return list(chars)

def mess_up_characters_in_words_list(prob, words_source):
    '''
    For every character in the words list, mess it up with a likelihood given by variable prob. 
    If a character is chosen to be messed up, map it into a randomly chosen character from the set of characters that appear in the list of words. 

    This function makes new list of words and left the words_source unchanged
    '''
    words = words_source.copy()
    characters = find_all_characters(words)
    for word_idx in range(len(words)):
        wl = list(words[word_idx])
        for i in range(len(wl)):
            if random.random() < prob:
                wl[i] = random.choice(characters)
        word = ''.join(wl)
        words[word_idx] = word
    return words

def find_all_words(words):
    '''
    Returns set of all distinct words that are in the given words sequence
    '''
    words_set = set()
    for word in words:
        words_set.add(word)
    return list(words_set)

def mess_up_words_in_words_list(prob, words_source):
    '''
    for every word in the text, mess it up with a likelihood given by variable prob. 
    If a word is chosen to be messed up, map it into a randomly chosen word from the set of words that appear in the text.

    This function makes new list of words and left the words_source unchanged
    '''
    words = words_source.copy()
    possible_words = find_all_words(words)
    for word_idx in range(len(words)):
        if random.random() < prob:
            words[word_idx] = random.choice(possible_words)
    return words

def run_one_experiment_characters(prob, words):
    '''
    Run 10 messing ups of characters with given probability and then compute the conditional entropy of the messed up text.
    Returns minimal, maximal and average conditional entropy 
    '''
    entropies = []
    for i in range(10):
        messed_words = mess_up_characters_in_words_list(prob, words)
        #print(messed_words)
        entropy = compute_conditional_entropy_of_words_list(messed_words)
        entropies.append(entropy)
    entropies = np.array(entropies)
    min_entropy = np.min(entropies)
    max_entropy = np.max(entropies)
    average_entropy = np.average(entropies)
    return (min_entropy, max_entropy, average_entropy)

def run_one_experiment_words(prob, words):
    '''
    Run 10 messing ups of words with given probability and then compute the conditional entropy of the messed up text.
    Returns minimal, maximal and average conditional entropy 
    '''
    entropies = []
    for i in range(10):
        messed_words = mess_up_words_in_words_list(prob, words)
        #print(messed_words)
        entropy = compute_conditional_entropy_of_words_list(messed_words)
        entropies.append(entropy)
    entropies = np.array(entropies)
    min_entropy = np.min(entropies)
    max_entropy = np.max(entropies)
    average_entropy = np.average(entropies)
    return (min_entropy, max_entropy, average_entropy)

def do_experiments(words):
    '''
    For given probabilities run the experiments described above
    '''
    probs = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0]
    char_entropies = []
    word_entropies = []
    for prob in probs:
        print(prob)
        char_entropies.append((prob, run_one_experiment_characters(prob, words)))
        word_entropies.append((prob, run_one_experiment_words(prob, words)))
    return char_entropies, word_entropies

def convert_result_entropy_to_perplexity(experiment_results):
    '''
    Converts the conditional entropy of experiments results to perplexity
    '''
    perplexities = []
    for res in experiment_results:
        entropies = res[1]
        perps = tuple()
        for ent in entropies:
            perp = compute_perplexity_from_entropy(ent)
            perps = perps + tuple([perp])
        perplexities.append((res[0], perps))
    return perplexities

def compute_total_number_of_characters(words):
    '''
    Returns number of characters in the given words sequence (excluding the whitespaces between words)
    '''
    total = 0
    for word in words:
        total += len(word)
    return total

def compute_number_of_words_with_given_frequency(words, freq):
    '''
    Returns number of words that appears in the text with given frequency
    '''
    words_freq = compute_words_occurence(words)
    num_of_words = 0
    for w, f in words_freq.items():
        if f == freq:
            num_of_words+=1
    return num_of_words


def basic_info_about_text(words):
    '''
    Prints basic information about given text (Results can be seen in the pdf file in tables in the first part of assignment)
    '''
    word_count = len(words)
    total_number_of_characters = compute_total_number_of_characters(words)
    average_number_of_characters = total_number_of_characters / word_count
    words_occur = compute_words_occurence(words)
    words_occur = dict(sorted(words_occur.items(), key= lambda x: x[1], reverse=True))
    tail = compute_number_of_words_with_given_frequency(words, 1)
    dif_words_count = len(words_occur)
    print("total words: {}".format(word_count))
    print("total number of characters: {}".format(total_number_of_characters))
    print("average number of chars per word: {}".format(average_number_of_characters))
    print(list(itertools.islice(words_occur.items(), 10)))
    print("Words with frequency 1: {}".format(tail))
    print("Number of differnet words: {}".format(dif_words_count))

#entropyCZ = compute_conditional_entropy_of_file("test.txt")
#perplexityCZ = compute_perplexity_from_entropy(entropyCZ)
#print(entropyCZ)
#print(perplexityCZ)

#entropyCZ = compute_conditional_entropy_of_file("TEXTCZ1.txt")
#perplexityCZ = compute_perplexity_from_entropy(entropyCZ)
#print(entropyCZ)
#print(perplexityCZ)
#
#entropyEN = compute_conditional_entropy_of_file("TEXTEN1.txt")
#perplexityEN = compute_perplexity_from_entropy(entropyEN)
#print(entropyEN)
#print(perplexityEN)

#words = load_words_from_file("test.txt")