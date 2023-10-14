import utils

cz_words = utils.load_words_from_file("TEXTCZ1.txt")
en_words = utils.load_words_from_file("TEXTEN1.txt")


# Pointwise mutual information of all the possible word pairs appearing consecutively in the data... :
cz_pmi = utils.compute_pointwise_mi(cz_words)
print("PMI CZ")
print(cz_pmi[:20])
print("Worst PMI CZ:", cz_pmi[-1])
print()

en_pmi = utils.compute_pointwise_mi(en_words)
print("PMI EN")
print(en_pmi[:20])
print("Worst PMI EN:", en_pmi[-1])
print()

# Pointwise mutual information of distant words (1<= distance <= 50)
cz_pmi_d = utils.compute_pointwise_mi_distant(cz_words)
print("PMI CZ DISTANT")
print(cz_pmi_d[:20])
print("Worst PMI CZ DISTANT:", cz_pmi_d[-1])
print()

en_pmi_d = utils.compute_pointwise_mi_distant(en_words)
print("PMI EN DISTANT")
print(en_pmi_d[:20])
print("Worst PMI EN DISTANT:", en_pmi_d[-1])

