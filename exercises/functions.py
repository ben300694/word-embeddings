from termcolor import colored
from tabulate import tabulate
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA  
from sklearn.manifold import TSNE    
import string
from gensim.parsing.preprocessing import remove_stopwords
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA

def illustrate_generation_of_data(input_text, architecture = "Skip-gram", window_size = 3):
    assert architecture in ["Skip-gram", "CBOW"], "Please select Skip-gram or CBOW as architecture"
    
    input_text_prepared = prepare_data(input_text)
    list_words = input_text_prepared.split()
    len_text = len(list_words)
    input_array = []
    output_array = []
    array_for_table = []
    
    for i in range(len_text - window_size*2 ):
        words_before = list_words[0:i]
        words_left_window = list_words[i:i+window_size]
        middle_word = list_words[i+window_size]
        words_right_window = list_words[i+window_size+1:i+2*window_size+1]
        words_after = list_words[i+2*window_size+1:]

        print(colored(" ".join(words_before), 'grey'), 
              colored(" ".join(words_left_window), 'red'), 
              colored(middle_word, 'green'), 
              colored(" ".join(words_right_window), 'red'), 
              colored(" ".join(words_after), 'grey'), "\n")
        
        if architecture == "Skip-gram":
            output_array.extend(words_left_window)
            output_array.extend(words_right_window)
            input_array.extend((window_size*2) * [middle_word])

            array_for_table_this_round = []
            for y in range(window_size*2+1):
                if y != window_size:
                    array_for_table_this_round.append((middle_word, list_words[i+y]))

            print(tabulate(array_for_table_this_round, headers=["Input", "Output"]), "\n")

            array_for_table.extend(array_for_table_this_round)
        
        elif architecture == "CBOW":
            output_array.extend([words_left_window + words_right_window])
            input_array.extend([middle_word])
            
            array_for_table_this_round = [(words_left_window + words_right_window, middle_word)]
            

            print(tabulate(array_for_table_this_round, headers=["Input", "Output"]), "\n")

            array_for_table.extend(array_for_table_this_round)
            
            
    return array_for_table

def prepare_data(input_text):
    input_text = input_text.lower()
    exclude = set(string.punctuation)
    output_text = ''.join(char for char in input_text if char not in exclude)
    return output_text

def show_table_of_training_data(data):
    print(tabulate(data, headers=["Input", "Output"]))

def visualize_vectors_by_colors(words, model, dim=100, figsize = (20, 5)):
    array_words = []
    for word in words:
        array_words.append(model.get_vector(word)[0:dim])
    fig, ax = plt.subplots(figsize=figsize)
    
    heatmap = plt.pcolor(array_words)
    
    plt.colorbar(heatmap, fraction=0.15, pad=0.04)

    plt.xticks(np.arange(dim)+0.5, np.arange(dim))
    plt.yticks(np.arange(len(words))+0.5, words, rotation='horizontal')
    
    ax.set_title("Visualization of word vectors by using different colors")
    plt.show()
    
def visualize_vectors_and_negative_by_color(words, model, dim=100, figsize=(20, 5)):
    array_words = []
    labels = []
    i = 0
    for word in words:
        array_words.append(model.get_vector(word)[0:dim])
        array_words.append(-model.get_vector(word)[0:dim])
        labels.append(words[i])
        labels.append(words[i]+'_negative')
        i += 1
    fig, ax = plt.subplots(figsize=figsize)
    
    heatmap = plt.pcolor(array_words)
    
    plt.colorbar(heatmap, fraction=0.15, pad=0.04)

    plt.xticks(np.arange(dim)+0.5, np.arange(dim))
    plt.yticks(np.arange(len(labels))+0.5, labels, rotation='horizontal')
    
    ax.set_title("Visualization of word vectors by using different colors")
    plt.show()

def print_string_arithmetics(positive, negative, words_most_similar):
    string_positive = ''
    string_negative = ''
    
    for i in range(len(positive)):
        if i == 0:
            string_positive += positive[i]
        else:
            string_positive += (' + ' + positive[i])
    
    for y in range(len(negative)):
        string_negative += (' - ' + negative[y])
    
    print(string_positive + string_negative, '=', *words_most_similar)

def tsne_out_of_words(words, model, dim):

    tsne_model = TSNE(perplexity=45, n_components=dim, init='pca', n_iter=3000, random_state=123)
    vectors = [model[word] for word in words]
    vectors_dim2 = tsne_model.fit_transform(vectors)
    return vectors_dim2

def plot_vectors_group_of_similar_words(vectors, number_groups, labels):
    
    assert number_groups < 8 and type(number_groups) == int, print('Please select an integer number between 1 and 6 as number of groups.')
    assert np.shape(vectors[1])[0] == 2, print('Your vectors need to have dimensionality of 2.')
    
    colors_total = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
    colors_here = colors_total[0:number_groups]
    number_per_group = int(len(vectors)/3)
    color_per_vector = []
    for i in range(len(colors_here)):
        color_per_vector.extend([colors_here[i] for j in range(number_per_group)])
    
    for j in range(number_groups):
        pos = number_per_group * j
        color_per_vector[pos] = 'y'
    
    plt.figure(figsize=(20, 10))
    
    x_values = vectors[:, 0]
    y_values = vectors[:, 1]
    
    plt.scatter(x_values, y_values, c = color_per_vector)
    
    for i in range(len(vectors)):
        plt.annotate(labels[i], xy = (x_values[i], y_values[i]))
    
    plt.title('Plot of word embeddings in lower dimensional space')
    plt.show()
    
def plot_tsne_of_words_similar_to_words_of_given_list(words, number_similar_per_word, model):
    words_total = [model]
    for word in words:
        words_total.append(word)
        words_similar_to_word = model.similar_by_word(word, number_similar_per_word)
        words_total.extend([w[0] for w in words_similar_to_word])
    words_total = words_total[1:]
        
    vectors_lower_dimension = tsne_out_of_words(words_total, model, 2)
    
    plot_vectors_group_of_similar_words(vectors_lower_dimension, len(words), words_total)
    
def print_examples_for_sentiments(data):
    indices_negative = [103, 1003]
    indices_neutral = [45, 146]
    indices_positive = [892, 3104]
    print('Example phrases for different sentiments:\n')
    
    print('Negative:\n')
    for index in indices_negative:
        print('-', data['Phrase'][index])
        
    print('\nNeutral:\n')
    for index in indices_neutral:
        print('-', data['Phrase'][index])

    print('\nPositive:\n')
    for index in indices_positive:
        print('-', data['Phrase'][index])
        
def clean_data(movie_review):
    #stopwords = ['and', 'or', 'then', 'i', 'you', 'what', 'when', 'then', 'we', 'that', 'on', 'or', 'the', 'is', 
    #            'of', 'a', 'with', 'for', 'its', 'it', 'the', 'an', 'for', 'by', 'his', 'at', 'from',
    #             'than', 'his', 'nt', 'about', 'one']
    movie_review = movie_review.lower()
    movie_review = BeautifulSoup(movie_review,'html.parser').get_text()
    movie_review_without_punctuation = movie_review.translate(str.maketrans('', '', string.punctuation))
    movie_review_without_stopwords = remove_stopwords(movie_review_without_punctuation)
    movie_review_wordlist = movie_review_without_stopwords.split()
    return movie_review_wordlist
    #movie_review_wordlist = movie_review_without_punctuation.split()
    #return [word for word in movie_review_wordlist if word not in stopwords]

def get_X_y_average_vector_for_each_phrase(phrases, labels, model, num_features, print_status=True):
    average_vectors = []
    indices_to_delete = []
    phrases_removed = []
    j = -1
    len_phrases = len(phrases)
    for phrase in phrases:
        j += 1
        i = 0
        vector = np.zeros(num_features)
        for word in phrase:
            if word in model.wv.index_to_key:
                i += 1
                vector += np.array(model.wv[word])
        if i == 0:
            indices_to_delete.append(j)
            phrases_removed.append(phrase)
        else:
            average_vectors.append(vector/i)
        if print_status:
            if(j%100000 == 0):
                print('{} out of {} phrases processed'.format(j, len_phrases))
    labels_without_nan = [x for y, x in enumerate(labels.tolist()) if y not in indices_to_delete]
    if print_status:
        print('Generation of Data successful')
    return average_vectors, labels_without_nan, phrases_removed


def normalise_row(row):
    if row['Sentiment'] == 0:
        return 0
    elif row['Sentiment'] == 1:
        return 0
    elif row['Sentiment'] == 2:
        return 1
    elif row['Sentiment'] == 3:
        return 2
    else:
        return 2
    
def classify_sample_review(review, clf, model, num_features):
    review = clean_data(review)
    average_vec,_,_ = get_X_y_average_vector_for_each_phrase([review], np.array([0]), model, num_features, print_status=False)
    prediction = int(clf.predict(average_vec)[0])
    if prediction == 0:
        pred = 'negative'
    elif prediction == 1:
        pred = 'neutral'
    elif prediction == 2:
        pred = 'positive'
    else:
        pred = 'not to classify'
    print("This review was classified as {}.".format(pred))

def plot_2d_pca(words, model):
    word_vectors = []
    for word in words:
        word_vectors.append(model[word])
    pca = PCA(n_components=2)
    pca.fit(word_vectors)
    Y = pca.transform(word_vectors)
    fig, ax = plt.subplots()
    ax.scatter(Y[:,0], Y[:,1])
    for i in range(len(words)):
        ax.annotate(words[i], (Y[i,0], Y[i,1]))
    plt.title('Plot of Word Vectors in 2d')
    plt.savefig('Plot_Relation',dpi=300)
    plt.show()


        

