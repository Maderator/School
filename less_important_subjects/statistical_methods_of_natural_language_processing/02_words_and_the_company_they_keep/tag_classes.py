import utils
import word_classes

def get_tag_class(words_and_tags, cl):
    '''
    Returns all words that appear with given tag cl
    '''
    class_wt = set()
    occ = 0
    for wt in words_and_tags:
        if wt[1] == cl:
            class_wt.add((wt[0], wt[1]))
            occ += 1
    print(occ)
    return class_wt

ent = utils.load_words_with_part_of_speech_tags("TEXTEN1.ptg")
print(len(ent))
tc1 = get_tag_class(ent, 'JJS')
print(tc1, len(tc1))
print()
tc2 = get_tag_class(ent, 'JJ')
print(tc2, len(tc2))


en = utils.get_tags_from_tags_list(ent)
#en = utils.load_words_from_file("TEXTEN1.txt")
history = word_classes.compute_full_class_hierarchy(en, word_classes=1, min_occurences=5)
print()
print(history)

#czt = utils.load_words_with_part_of_speech_tags("TEXTCZ1.ptg")
#cz = utils.get_tags_from_tags_list(czt)
##cz = utils.load_words_from_file("TEXTCZ1.txt")
#cz8000 = cz[:8000]
#history = word_classes.compute_full_class_hierarchy(cz8000, n=15)
#print(history)