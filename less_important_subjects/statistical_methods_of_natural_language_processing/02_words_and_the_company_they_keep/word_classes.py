import utils
import math
import numpy as np
import copy

class WordClass:
    '''
    Word class used in the word class algorithm
    '''

    def __init__(self, class_id, initial_words, occurences):
        '''
        input:
            class_id - Int used as an identificator of this class
            initial_words - Set of initial words stored in this word class
            occurences - Number of occurences of the given initial word in the text
        '''
        self.class_id = class_id # We use int as a id
        self.words = initial_words   # Set of initial words of this class
        self.occurences = occurences
        self.merge_history = []

    def merge_with_another_class(self, another_class):
        ''' Merge another class words with this class words '''
        self.words.extend(another_class.words)
        self.occurences += another_class.occurences
        self.merge_history.append(copy.deepcopy(another_class))   
    
    def ___str__(self):
        return "(" + str(self.words) + str(self.class_id) + ")"

def get_words_without_tags(tag_words):
    '''
    Throw away tags and return list of words in the same order as the tuples (word, tag) but without tag
    '''
    words = []
    for w, t in tag_words:
        words.append(w)
    return words

def get_tag_words_occuring_n_times_or_more(tag_words, n=10):
    words = get_words_without_tags(tag_words)
    occ = utils.compute_words_occurence(words)
    new_tw = set() # tag words that occur n times or more
    for w, t in tag_words:
        if occ[w] >= n:
            new_tw.add((w, t))
    return new_tw

# TODO use the bigram count table in lecture slide 124 45 minutes 27/10 12:20
def compute_pointwise_mi_of_classes(log_subterms):
    '''
    Computes pointwise mutual information for all the possible class pairs
    appearing consecutively in the data.
    '''
    mi = 0
    for q, val in log_subterms.items():
        mi += val
    return mi

def count_number_of_bigrams(bigram_counts):
    n_of_bigrams = 0
    for b, c in bigram_counts.items():
        n_of_bigrams += c
    return n_of_bigrams

def init_bigram_table(classes, bigram_counts):
    '''
    Initializes table of bigrams for each class.
    inputs:
        classes - given classes for which we will create the table
        bigram_counts - bigram counts for bigrams consisting of words from the text
    '''
    nc = len(classes)
    bi_table = {} # dictionary of two classes (cl1, cl2) and the given count
    for cl1 in classes:
        for cl2 in classes:
            count_sum = 0
            for w1 in cl1.words:
                for w2 in cl2.words:
                    bi = (w1, w2)
                    if bi in bigram_counts:
                        count_sum += bigram_counts[bi]
            bi_table[(cl1, cl2)] = count_sum
    return bi_table

## Functions for initialization phase of word classes algorithm

def init_class_unigram_counts(words, classes, bigram_table):
    '''
    Initialize unigram counts for classes
    returns
        l_count - unigram count for classes on left side of bigrams
        r_count - unigram count for classes on right side of bigrams
    '''
    l_count, r_count = {}, {}
    cn = len(classes)
    for cur_class in classes:
        lc, rc = 0, 0 # count for left and right class unigram
        for other_class in classes:
            lb = (cur_class, other_class) # bigram with cur class on left
            rb = (other_class, cur_class) # bigram with cur class on right
            lc += bigram_table[lb] # left count
            rc += bigram_table[rb] # right count
        l_count[cur_class] = lc
        r_count[cur_class] = rc
    return l_count, r_count

def compute_q_ab_r(cla, clb, clr, l_uni_counts, r_uni_counts, c_bigram_table, wc):
    '''
    Macro for computation of q with a and b classes being merged together. 
    q_k(a+b,r)
    '''
    par = c_bigram_table[(cla,clr)]/wc
    pbr = c_bigram_table[(clb,clr)]/wc
    plr = par + pbr
    uni_pa = l_uni_counts[cla]/wc
    uni_pb = l_uni_counts[clb]/wc
    uni_pl = uni_pa + uni_pb
    uni_pr = r_uni_counts[clr]/wc
    if plr == 0:
        return 0
    else:
        return plr * math.log2(plr / (uni_pl * uni_pr))

def compute_q_l_ab(cll, cla, clb, l_uni_counts, r_uni_counts, c_bigram_table, wc):
    '''
    Macro for computation of q with a and b classes being merged together. 
    q_k(l,a+b)
    '''
    pla = c_bigram_table[(cll,cla)]/wc
    plb = c_bigram_table[(cll,clb)]/wc
    plr = pla + plb
    uni_pl = l_uni_counts[cll]/wc
    uni_pa = r_uni_counts[cla]/wc
    uni_pb = r_uni_counts[clb]/wc
    uni_pr = uni_pa + uni_pb
    if plr == 0:
        return 0
    else:
        return plr * math.log2(plr / (uni_pl * uni_pr))

def compute_q_ab_ab(cla, clb, l_uni_counts, r_uni_counts, c_bigram_table, wc):
    '''
    Macro for computation of q with a and b classes being merged together. 
    q_k(a+b,a+b)
    '''
    paa = c_bigram_table[(cla,cla)]/wc
    pab = c_bigram_table[(cla,clb)]/wc
    pba = c_bigram_table[(clb,cla)]/wc
    pbb = c_bigram_table[(clb,clb)]/wc
    plr = paa + pab + pba + pbb
    uni_pla = l_uni_counts[cla]/wc
    uni_plb = l_uni_counts[clb]/wc
    uni_pl = uni_pla + uni_plb
    uni_pra = r_uni_counts[cla]/wc
    uni_prb = r_uni_counts[clb]/wc
    uni_pr = uni_pra + uni_prb
    if plr == 0:
        return 0
    else:
        return plr * math.log2(plr / (uni_pl * uni_pr))

def compute_log_subterms(classes, l_uni_counts, r_uni_counts, c_bigram_table, word_count):
    '''
    Initialize subterms involving a log
    '''
    log_subterms = {}

    for cl1 in classes:
        for cl2 in classes:
            cl_bigram = (cl1, cl2)
            bic = c_bigram_table[cl_bigram]
            ulc = l_uni_counts[cl1]
            urc = r_uni_counts[cl2]
            if bic == 0:
                log_subterms[cl_bigram] = 0
            else:
                log_subterms[cl_bigram] = bic/word_count * math.log2(word_count*bic/(ulc * urc))
    print("mutual information:", compute_pointwise_mi_of_classes(log_subterms))

    return log_subterms

def compute_subtraction_subterms(all_classes, merging_classes, log_subterms):
    '''
    Initialize "subtraction" subterms
    '''
    subterms = {}
    for mc in merging_classes:
        l_ls_sum = 0 # log subterm sum over left classes with right class being cl
        r_ls_sum = 0 # log subterm sum over right classes with left class being cl
        for other_class in all_classes:
            l_ls_sum += log_subterms[(other_class, mc)]
            r_ls_sum += log_subterms[(mc, other_class)]
        subterms[mc] = l_ls_sum + r_ls_sum - log_subterms[(mc, mc)]
    return subterms

def compute_loss(cl1, cl2, all_classes, subterms, log_subterms, l_uni_c, r_uni_c, bigram_table, wc):
    si = subterms[cl1] 
    sj = subterms[cl2]
    qab = log_subterms[(cl1, cl2)]
    qba = log_subterms[(cl2, cl1)]

    qabab = compute_q_ab_ab(cl1, cl2, l_uni_c, r_uni_c, bigram_table, wc)

    sums = 0
    for cl3 in all_classes:
        if cl3 != cl1 and cl3 != cl2:
            sums += compute_q_l_ab(cl3, cl1, cl2, l_uni_c, r_uni_c, bigram_table, wc)
            sums += compute_q_ab_r(cl1, cl2, cl3, l_uni_c, r_uni_c, bigram_table, wc)

    loss = si + sj - qab - qba - qabab - sums
    return loss

def init_table_of_losses(all_classes, merging_classes, subterms, log_subterms_table, l_uni_c, r_uni_c, bigram_table, wc):
    '''
    Initialize upper-right triangle table of losses
    '''
    min_loss = float("inf")
    a, b = None, None
    losses = {}
    for id1, cl1 in enumerate(merging_classes):
        for id2, cl2 in enumerate(merging_classes):
            if id2 > id1:
                losses[(cl1,cl2)] = compute_loss(cl1, cl2, all_classes, subterms, log_subterms_table, l_uni_c, r_uni_c, bigram_table, wc)
                if losses[(cl1,cl2)] < min_loss:
                    min_loss = losses[(cl1, cl2)]
                    a = cl1
                    b = cl2
    return losses, a, b, min_loss

## Select and update functions

def merge_two_classes_counts(ca, cb, all_classes, bi_table):
    '''
    Given the two classes ca and cb, merge the bigram table counts
    '''
    for cl1 in all_classes:
        if cl1 != ca and cl1 != cb:
            bi_table[(ca, cl1)] += bi_table[(cb, cl1)]
            del bi_table[(cb, cl1)]
            bi_table[(cl1, ca)] += bi_table[(cl1, cb)]
            del bi_table[(cl1, cb)]
    bi_table[(ca,ca)] = bi_table[(ca,ca)] + bi_table[(ca,cb)] + bi_table[(cb,ca)] + bi_table[(cb,cb)]
    del bi_table[(ca,cb)]
    del bi_table[(cb,ca)]
    del bi_table[(cb,cb)]

def merge_classes(all_classes, merging_classes, merge_history, ca, cb, l_uni_c, r_uni_c, bi_table):
    '''
    Merge two classes together and also the bigram and unigram counts of the two classes
    '''
    merge_history.append((ca.class_id, cb.class_id))
    
    merge_two_classes_counts(ca, cb, all_classes, bi_table)
    l_uni_c[ca] += l_uni_c[cb]
    r_uni_c[ca] += r_uni_c[cb]
    del l_uni_c[cb]
    del r_uni_c[cb]
    
    ca.merge_with_another_class(cb)
    all_classes.remove(cb)
    merging_classes.remove(cb)

def update_table_of_losses(
    old_losses_table, all_classes, merging_classes, 
    ca, cb, subterms, new_subterms, log_subterms, new_log_subterms,
    l_uni_c, r_uni_c, c_bi_table,
    old_l_uni_c, old_r_uni_c, old_c_bi_table, nw): # ca, cb are classes causing minimal loss of MI
    '''
    Updates table of losses. Based on the equations from lecture presentation
    https://ufal.mff.cuni.cz/~pecina/courses/npfl067/slides/npfl067-08.pdf
    slide 131
    '''
    min_loss = float("inf")
    a, b = None, None
    losses = {}
    for id1, cl1 in enumerate(merging_classes):
        for id2, cl2 in enumerate(merging_classes):
            if id2 > id1:
                if cl1 != cb and cl2 != cb:
                    if cl1 != ca and cl2 != ca:
                        losses[(cl1, cl2)] = (old_losses_table[(cl1, cl2)] 
                                            - subterms[cl1] + new_subterms[cl1] 
                                            - subterms[cl2] + new_subterms[cl2]
                                            + compute_q_ab_r(cl1, cl2, ca, old_l_uni_c, old_r_uni_c, old_c_bi_table, nw) + compute_q_l_ab(ca, cl1, cl2, old_l_uni_c, old_r_uni_c, old_c_bi_table, nw)
                                            + compute_q_ab_r(cl1, cl2, cb, old_l_uni_c, old_r_uni_c, old_c_bi_table, nw) + compute_q_l_ab(cb, cl1, cl2, old_l_uni_c, old_r_uni_c, old_c_bi_table, nw)
                                            - compute_q_ab_r(cl1, cl2, ca, l_uni_c, r_uni_c, c_bi_table, nw) - compute_q_l_ab(ca, cl1, cl2, l_uni_c, r_uni_c, c_bi_table, nw))
                        losses[(cl1, cl2)] = old_losses_table[(cl1,cl2)]
                    else: # cl1 or cl2 is previously merged class ca
                        losses[(cl1, cl2)] = compute_loss(cl1, cl2, all_classes, new_subterms, new_log_subterms, l_uni_c, r_uni_c, c_bi_table, nw)
                    if losses[(cl1,cl2)] < min_loss:
                        min_loss = losses[(cl1, cl2)]
                        a = cl1
                        b = cl2
    return losses, a, b, min_loss

## Main function
def create_n_word_classes(words, w_mapping, merging_classes, merging_words, all_classes, n=1):
    '''
    Iteratively merges initial_classes until we are left with n classes. 
    inputs:
        words - Words from which we get the bigram counts and other training data
        w_mapping - mapping of words to classes
        merging_classes - Set of initial classes of words from trainig "words" that are to be merged
        merging_words - Words that are being merged in merging_classes
        all_classes - Set of all classes of words including those words from training "words" that are not going to be merged
        n - final number of classes in merging_classes
    returns:
        1. final shape of merged classes
        2. history of merges
        3. initial shape of class being merged
    '''
    merge_history = []
    initial_merging_classes = copy.deepcopy(merging_classes)
    nc = len(merging_classes) # number of classes
    iterations = nc - n
    if iterations < 0:
        print("Given n is bigger than number of initial classes. Returning initial classes.")
        return merging_classes, merge_history

    ## INITIALIZATION PHASE ##
    
    w_unigram_counts = utils.compute_unigram_count(words) # being careful at the beginning of the text
    # 1 (Read data in,) init bigram count table
    w_bigram_counts = utils.compute_bigram_count(words)
    nw = count_number_of_bigrams(w_bigram_counts)

    c_bigram_table = init_bigram_table(all_classes, w_bigram_counts)
    
    # 2 Init unigram counts
    l_uni_c, r_uni_c = init_class_unigram_counts(words, all_classes, c_bigram_table)
    
    # 3 Init subterms q involving a log 
    log_subterms_table = compute_log_subterms(all_classes, l_uni_c, r_uni_c, c_bigram_table, nw)
    
    # 4 Init "subtraction" subterms s
    subterms = compute_subtraction_subterms(all_classes, merging_classes, log_subterms_table)

    # 5 Init table of losses L (watch candidates for selection of best pair for merge)
    c_table_of_losses, ca, cb, min_loss = init_table_of_losses(all_classes, merging_classes, subterms, log_subterms_table, l_uni_c, r_uni_c, c_bigram_table, nw)
    print("Minimal loss:", min_loss, "for", ca.words, "+", cb.words)

    ## SELECT PHASE ##
    # 6 Select the best pair of classes (a,b) to merge into class a 
    #   (watch the candidates when computing table of losses L_k(a,b))    
    #ca, cb = select_best_classes()
    #   Save selection to history
    old_l_uni_c = l_uni_c.copy()
    old_r_uni_c = r_uni_c.copy()
    old_c_bigram_table = c_bigram_table.copy()

    merge_classes(all_classes, merging_classes, merge_history, ca, cb, l_uni_c, r_uni_c, c_bigram_table)

    ## UPDATE PHASE ##
    for i in range(iterations-1): # one merging was already done so we subtract iterations by 1

        # 7 Optionally, update q_k(i,j) to get q_{k-1}(i,j)
        new_log_subterms_table =  compute_log_subterms(all_classes, l_uni_c, r_uni_c, c_bigram_table, nw)
        
        # 8 Optionally, update s_k(i) to get s_{k-1}(i)
        new_subterms =            compute_subtraction_subterms(all_classes, merging_classes, new_log_subterms_table)
            
        # 9 Update the loss table
        c_table_of_losses, ca, cb, min_loss = update_table_of_losses(c_table_of_losses, all_classes, merging_classes, ca, cb,
            subterms, new_subterms, log_subterms_table, new_log_subterms_table, 
            l_uni_c, r_uni_c, c_bigram_table, 
            old_l_uni_c, old_r_uni_c, old_c_bigram_table, nw) # ca, cb are classes causing minimal loss of MI
        print("Minimal loss:", min_loss, "for", ca.words, "+", cb.words)
        # 10 During the loss table update, keep track of the minimal loss of MI, and the two classes which cause it.
        # 11 Remember such best merge in merge_history
        old_l_uni_c = l_uni_c.copy()
        old_r_uni_c = r_uni_c.copy()
        old_c_bigram_table = c_bigram_table.copy()
        merge_classes(all_classes, merging_classes, merge_history, ca, cb, l_uni_c, r_uni_c, c_bigram_table)
        
        # 12 get rid of all subterms, log_subterms and table of losses from previous iteration
        log_subterms_table = new_log_subterms_table
        subterms = new_subterms

    return merging_classes, merge_history, initial_merging_classes

def create_initial_classes(words, n=10):
    '''
    Initialize classes where each class has one word with n or more occurences in text (words).
    returns mapping of words to classes and the classes themselves
    '''
    uni = utils.compute_unigram_count(words)
    merging_classes = set()
    merging_words = set()
    all_classes = set()
    word_mapping = {}
    i = 0
    for w, oc in uni.items():
        new_class = WordClass(i, [w], oc)
        all_classes.add(new_class)
        word_mapping[w] = i
        if oc >= n: # ten or more occurences of word in text
            merging_classes.add(new_class)
            merging_words.add(w)
        i += 1
    return word_mapping, merging_classes, merging_words, all_classes

def print_classes_and_their_id(classes):
    '''
    Print sorted classes with only one word and their tag. 
    '''
    cl_tuples = []
    for cl in classes:
        cl_tuples.append((cl.words[0], cl.class_id))
    cl_tuples_sorted = sorted(cl_tuples, key=lambda x: x[1])
    for cl in cl_tuples_sorted:
        print("(" + str(cl[0]) + "," + str(cl[1]) + ")", end=', ')

def compute_full_class_hierarchy(words, word_classes=1, min_occurences=10):
    '''
    "Main" function which computes the full class hierarchy and prints main info about classes.
    '''
    word_mapping, merging_classes, merging_words, all_classes = create_initial_classes(words, n=min_occurences)
    print(len(all_classes))
    print("Initial classes:")
    print_classes_and_their_id(merging_classes)
    final_classes, history, initial_classes = create_n_word_classes(
        words, word_mapping, merging_classes, merging_words, all_classes, n=word_classes)
    print("initial classes:", len(initial_classes))
    print("all classes:", len(all_classes))
    print("\nFinal classes")
    print_classes_and_their_id(final_classes)
    return history

if __name__ == "__main__":
    #ent = utils.load_words_with_part_of_speech_tags("TEXTEN1.ptg")
    #en = utils.get_words_from_tags_list(ent)
    ##en = utils.load_words_from_file("TEXTEN1.txt")
    #en8000 = en[:8000]
    #history = compute_full_class_hierarchy(en8000, word_classes=15, min_occurences=10)
    #print(history)

    czt = utils.load_words_with_part_of_speech_tags("TEXTCZ1.ptg")
    cz = utils.get_words_from_tags_list(czt)
    #cz = utils.load_words_from_file("TEXTCZ1.txt")
    cz8000 = cz[:8000]
    history = compute_full_class_hierarchy(cz8000, word_classes=15, min_occurences=10)
    print(history)