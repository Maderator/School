import jsonlines
import re

def omit_words_of_length(length, sentence):
    words = sentence.split()
    new_words = []
    for word in words:
        if len(word) > length:
            new_words.append(word)
    return ' '.join(new_words)

def create_ingredients_set(recipes):
    set_ing = set()
    for obj in recipes:
        for ing in obj["ingredients"]:
            words = ing.split()
            for word in words:
                set_ing.add(word)
    return set_ing

def remove_s_es_from_ingredients(sentence, ingredients_dict):
    words = sentence.split()
    new_words = []
    for word in words:
        if word[-1] == 's':
            if word[-2] == 'e':
                if word[:-2] in ingredients_dict:
                    new_words.append(word[:-2])
                elif word[:-1] in ingredients_dict:
                    new_words.append(word[:-1])
                else:
                    new_words.append(word)
            else:
                if word[:-1] in ingredients_dict:
                    new_words.append(word[:-1])
                else:
                    new_words.append(word)
            
        else:
            new_words.append(word)

    return ' '.join(new_words)

def remove_suffixes_from_recipes(cleaned_recipes, all_ingredients):
    for i in range(len(cleaned_recipes)):
        rec_ings = cleaned_recipes[i]['ingredients']
        for ing_idx, ing in enumerate(rec_ings): 
            rec_ings[ing_idx] = remove_s_es_from_ingredients(ing, all_ingredients)
        cleaned_recipes[i]['ingredients'] = rec_ings
    return cleaned_recipes
        
def get_frequent_words(recipes, min_frequency):
    words_frequency = dict()
    for rec in recipes:
        ings = rec['ingredients']
        for ing in ings:
            words = ing.split()
            for word in words:
                if word in words_frequency:
                    words_frequency[word] = words_frequency[word]+1
                else:
                    words_frequency[word] = 1
    
    frequent_words = list()
    for w, freq in words_frequency.items():
        if freq >= min_frequency:
            frequent_words.append(w)
    return frequent_words

def remove_stopwords_from_ingredient(ing, stopwords):
    words = ing.split()
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return ' '.join(new_words)

def remove_stopwords(cleaned_recipes, stopwords):
    for i, rec in enumerate(cleaned_recipes):
        ings = rec['ingredients']
        for ing_idx, ing in enumerate(ings):
            ings[ing_idx] = remove_stopwords_from_ingredient(ing, stopwords)
        cleaned_recipes[i]['ingredients'] = ings
    return cleaned_recipes

def process_recipes(file):
    alpha_lower_regex = re.compile('[^a-z ]')
    recipes_file = file

    cleaned_recipes = []
    with jsonlines.open(recipes_file, mode='r') as reader:
        for obj in reader:
            #print("Name:", obj["name"])
            #print("Ingredients:")
            to_remove = []
            for i in range(len(obj["ingredients"])):
                #print("   ", obj["ingredients"][i])
                ing = obj["ingredients"][i]

                ing = ing.lower() # 2. Convert all letters into lower case lettersremove all non-alphabetic characters. 
                ing = alpha_lower_regex.sub('', ing) # 2. remove all non-alphabetic characters. 

                ing = omit_words_of_length(2, ing)
                if len(ing) <= 2: # 3. omit all words of length at most two.
                    to_remove.append(ing)

                obj["ingredients"][i] = ing
            for el in to_remove: 
                obj["ingredients"].remove(el)
            cleaned_recipes.append(obj)
        all_ingredients = create_ingredients_set(cleaned_recipes)

    cleaned_recipes = remove_suffixes_from_recipes(cleaned_recipes, all_ingredients)

    frequent_words = get_frequent_words(cleaned_recipes, 30)

    #print(frequent_words)
    stopwords = ['pound', 'piece', 'about', 'used', 'extra', 'virgin', 'finely', 'chopped',
        'fresh', 'minced', 'whole', 'freshly', 'grated', 'divided', 'romano', 'flat',
        'leaf', 'beaten', 'kosher', 'white', 'that', 'cut',
        'into', 'large', 'cube', 'see', 'you', 'will', 'heavy', 'pinch', 'additional',
        'cup', 'stick', 'ounce', 'box', 'baby', 'black', 'red', 'sharp', 
        'shredded', 'with', 'one', 'teaspoon', 'allpurpose', 'plus', 'more', 'needed', 'package', 'dry', 'melted', 'few',
        'granulated', 'tablespoon', 'softened', 'can', 'golden', 'mix', 'and', 'for', 'approximately', 'weight', 'grind', 'each',
        'cored', 'from', 'use', 'sweet', 'sliced', 'skin', 'squeezed', 'the', 'two', 'small',
        'thinly', 'shortening', 'inch', 'pan', 'italian', 'removed', 'casing', 'diced', 'half',
        'green', 'flake', 'bunch', 'note', 'below', 'coarsely', 'thick', 'clove', 
        'sprig', 'bay', 'leave', 'drained', 'cooked', 'but', 'left', 'such', 
        'crushed', 'boneless', 'skinless', 'slice', 'dried', 'garnish', 'yellow', 'light', 
        'unsalted', 'taste', 'cold', 'leftover', 'medium', 'stalk', 'quartered', 'uncooked', 'sage', 'fine', 'frozen', 
        'top', 'zest', 'very', 'our', 'bite', 'sized', 'head', 'vidalia', 'also', 'raw', 'peeled', 'your', 'seasoning', 
        'only', 'thin', 'brand', 'frying', 'semisweet', 
        'dijon', 'cleaned', 'hot', 'not', 'favorite', 'good', 'quality', 'unsweetened', 'extravirgin', 'thigh', 'trimmed', 
        'crumbs', 'cooking', 'strip', 'any', 'seed', 
        'all', 'packed', 'like', 'optional', 'serving', 'room', 'temperature', 'dark', 'other', 'seeded', 'dice', 
        'recipe', 'this', 'canned', 'ripe', 'sea', 'holes', 'grater', 'prepared', 'jack', 'above', 'quarter', 'stem', 
        'available', 'supermarket', 'chip', 'stemmed', 'spray', 'jarred', 'organic', 
        'ground', 'brown', 'skinned', 'shelled', 'roasted', 'vegetable', 'some', 'try', 'find', 'vertical', 'end', 'key', 'ingredient', 'should', 'omitted', # my additional words
        'may', 'substituted', 'firm', 'counter', 'dusting', 'separated', 'being'] # my additional words

    cleaned_recipes = remove_stopwords(cleaned_recipes, stopwords)

    #print(cleaned_recipes)    

    #with jsonlines.open('recipes_processed.jl', mode='w') as writer:
    #    for rec in cleaned_recipes:
    #        writer.write(rec)

    return cleaned_recipes

        