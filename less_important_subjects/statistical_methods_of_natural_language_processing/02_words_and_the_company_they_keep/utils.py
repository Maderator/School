import math

def load_words_from_file(file):
    with open(file, 'r', encoding='iso-8859-2') as reader:
        words = [line.strip('\n') for line in reader.readlines()] # load words from file without endline symbol
    return words

def load_words_with_part_of_speech_tags(file):
    with open(file, 'r', encoding='iso-8859-2') as reader:
        words_and_tags = [line.strip('\n').split('/') for line in reader.readlines()] # load words from file without endline symbol
    return words_and_tags

def get_words_from_tags_list(words_and_tags):
    words = []
    for w, t in words_and_tags:
        words.append(w)
    return words

def get_tags_from_tags_list(words_and_tags):
    tags = []
    for w,t in words_and_tags:
        tags.append(t)
    return tags

def add_unigram(word, uni_counts):
    '''
    add the word to unigram counts or if it is already in the uni. counts, increment the counter
    '''
    if word in uni_counts:
        uni_counts[word] += 1
    else:
        uni_counts[word] = 1

def compute_unigram_count(words):
    '''
    Count unigram occurences
    '''
    unigram_counts = {}
    for i in range(1, len(words)): # We are carefull at the beginning of the text
        add_unigram(words[i], unigram_counts)
    return unigram_counts

def add_bigram(word1, word2, bi_counts):
    '''
    creates bigram from given two words and add it to the bigram dictionary or increment the count
    '''
    bigram = (word1, word2)
    if bigram in bi_counts:
        bi_counts[bigram] += 1
    else:
        bi_counts[bigram] = 1

def compute_bigram_count(words):
    '''
    Count bigrams occurences
    '''
    bigram_counts = {}
    for i in range(len(words)-1):
        word1 = words[i]
        word2 = words[i+1]
        add_bigram(word1, word2, bigram_counts)
    return bigram_counts

def compute_bigram_count_distant(words):
    '''
    Count bigrams occurences of words that are at least 1 word appart, but not farther than 50 words (both directions)
    '''
    bigram_counts = {}
    for i in range(len(words)-1):
        j = i-2
        while j >= 0 and i-1-j <= 50:
            add_bigram(words[i], words[j], bigram_counts)
            j -= 1
        j = i+2
        while j < len(words) and j-1-i <= 50:
            add_bigram(words[i], words[j], bigram_counts)
            j+=1
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

def compute_words_occurence_probability(words):
    '''
    Count the probability of word in given words sequence
    ''' 
    words_occurs = compute_words_occurence(words)
    words_prob = words_occurs
    total_words = len(words)
    for w, count in words_prob.items():
        words_prob[w] = count / total_words
    return words_prob

def compute_joint_prob(bigram_count, number_of_words):
    ''' 
    return dictionary with probability P(i,j) that at any position in the text you will find the word i followed immediately by the word j
    '''
    probability = bigram_count
    for k, v in probability.items():
        probability[k] = v/number_of_words
    return probability

def compute_conditional_prob(joint_prob, word_prob):
    '''
    joint_prob - precomputed joint probability 
    word_prob  - precomputed probability of word occurence probability in text
    return dictionary with probability P(j|i) that if word i occurs in the text then word j will follow
    '''
    cond_prob = joint_prob.copy()
    for bi, joint_p in cond_prob.items():
        cond_prob[bi] = joint_p / word_prob[bi[0]]
    return cond_prob

def has_n_or_more_occurences(word, unigram_counts, n=10):
    if word in unigram_counts:
        if unigram_counts[word] >= n:
            return True
    return False

def compute_pointwise_mi(words, n=10):
    '''
    Computes pointwise mutual information for all the possible word pairs
    appearing consecutively in the data, disregarding pairs in which one
    or both words appear less than n times in the corpus.
    '''
    uni = compute_unigram_count(words)
    wop = compute_words_occurence_probability(words)
    bic = compute_bigram_count(words)
    bic_ten = {}
    for b, v in bic.items(): 
        # disregarding pairs in which one or both words appear less than 10 times in corpus
        if has_n_or_more_occurences(b[0], uni, 10) and has_n_or_more_occurences(b[1], uni, 10):
            bic_ten[b] = v
    jp = compute_joint_prob(bic_ten, len(words))
    #cp = compute_conditional_prob(jp, wop)
    
    mi = []
    for bi, j_prob in jp.items():
        mutual_inf = math.log2(j_prob / (wop[bi[0]] * wop[bi[1]]))
        mi.append((bi,mutual_inf))

    mi_s = sorted(mi, key=lambda x: x[1], reverse=True)
    return mi_s

def compute_pointwise_mi_distant(words, n=10):
    '''
    Computes pointwise mutual information for all the possible pairs of distant words, 
    i.e. words which are at least 1 word apart, but not farther than 50 words (both directions).
    Disregarding pairs in which one or both words appear less than n times in the corpus.
    '''
    uni = compute_unigram_count(words)
    wop = compute_words_occurence_probability(words)
    bic = compute_bigram_count_distant(words)
    bic_ten = {}
    for b, v in bic.items(): # disregarding pairs in which one or both words appear less than 10 times in corpus
        if has_n_or_more_occurences(b[0], uni, 10) and has_n_or_more_occurences(b[1], uni, 10):
            bic_ten[b] = v
    jp = compute_joint_prob(bic_ten, len(words))

    mi = []
    for bi, j_prob in jp.items():
        mutual_inf = math.log2(j_prob / (wop[bi[0]] * wop[bi[1]]))
        mi.append((bi,mutual_inf))

    mi_s = sorted(mi, key=lambda x: x[1], reverse=True)
    return mi_s

