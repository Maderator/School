import entropy as ent
import matplotlib.pyplot as plt
import numpy as np

'''
File with experiments from first part of the assignment 1
(Second part with language modeling and cross entropy is in file language_modeling.py)
'''

def convert_to_plot_format(data):
    '''
    Convert data so that they can be plotted
    '''
    evals = []
    lower = []
    mean = []
    upper = []
    for d in data:
        evals.append(d[0])    
        lower.append(d[1][0])
        upper.append(d[1][1])
        mean.append(d[1][2])
    return evals, lower, upper, mean

def plot_experiment(data, legend_name=''):
    evals, lower, upper, mean = convert_to_plot_format(data)
    plt.plot(evals, mean, label=legend_name)
    plt.fill_between(evals, lower, upper, alpha=0.25)

wordsCZ = ent.load_words_from_file("test.txt")
entropy_charCZ, entropy_wordCZ = ent.do_experiments(wordsCZ)
#plot_experiment(entropy_charCZ, legend_name='Entropy Czech Characters')
#plot_experiment(entropy_wordCZ, legend_name='Entropy Czech Words')

perplexity_charCZ = ent.convert_result_entropy_to_perplexity(entropy_charCZ)
perplexity_wordCZ = ent.convert_result_entropy_to_perplexity(entropy_wordCZ)
plot_experiment(perplexity_charCZ, legend_name='Perplexity Czech Characters')
plot_experiment(perplexity_wordCZ, legend_name='Perplexity Czech Words')

print("CZECH TEXT DATA:")
ent.basic_info_about_text(wordsCZ)
print(entropy_charCZ)
print(perplexity_charCZ)
print()
print(entropy_wordCZ)
print(perplexity_wordCZ)

print()
wordsEN = ent.load_words_from_file("TEXTEN1.txt")
entropy_charEN, entropy_wordEN = ent.do_experiments(wordsEN)
#plot_experiment(entropy_charEN, legend_name='Entropy English Characters')
#plot_experiment(entropy_wordEN, legend_name='Entropy English Words')

perplexity_charEN = ent.convert_result_entropy_to_perplexity(entropy_charEN)
perplexity_wordEN = ent.convert_result_entropy_to_perplexity(entropy_wordEN)
plot_experiment(perplexity_charEN, legend_name='Perplexity English Characters')
plot_experiment(perplexity_wordEN, legend_name='Perplexity English Words')

print("ENGLISH TEXT DATA:")
ent.basic_info_about_text(wordsEN)
print(entropy_charEN)
print(perplexity_charEN)
print()
print(entropy_wordEN)
print(perplexity_wordEN)

plt.title("Perplexity of text")
plt.xlabel("Probability of messing up")
plt.ylabel("Perplexity")
plt.legend(loc="lower left")
#plt.yscale('log')
#plt.loglog()
plt.show()