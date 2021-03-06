from PyDictionary import PyDictionary
import time
import os
import numpy as np
import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd
import itertools
from sklearn.model_selection import KFold


# Webpage object (TCRF)
# Reponse methods found here: https://www.geeksforgeeks.org/response-methods-python-requests/
class Webpage:
    def __init__(self, url):
        self.url = url
        self.response = requests.get(url)
        self.dom = BeautifulSoup(self.response.text, 'html.parser')


class feature_functions:
    def __init__(self):
        self.funcdict = np.array([
            self.all_uppercase_function, 
            self.first_uppercase_function,
            self.is_photo_function,
            self.is_positive_word,
            self.is_english_word
        ])
        

        self.epochs = 30
        self.batch_size = 100
        self.weights = np.array(np.random.uniform(low = 0.5, high = 10.0, size = (len(self.funcdict))).tolist(), dtype=np.float128)
        self.alpha = 1.0
        self.n_feature_functions = len(self.funcdict)
        self.all_labels = ["standard word", "name", "pic", "position", "affiliation", "phone", "address", "email", "Phddate", "Phduniv", "Phdmajor", "Msdate", "Msuniv",
"Msmajor", "Bsdate", "Bsuniv", "Bsmajor", "contactinfo" ]
        self.stack = ['standard word']

        print("Initialized")


    # PROBABILITY CALCULATIONS (FOR PREDICTION)
    # P(label sequence | sentence) = exp(sum(w * sum(feature_function(labels, sentence)))) / normalization constant
    def p_theta(self, sentence, labels):
        if type(sentence) == str:
            splitted = sentence.split()
        else: 
            splitted = sentence
        all_label_combos = self.all_label_combinations(len(labels)) # get all combinations and permutations of labels

        numerator = self.p_theta_numerator(splitted, labels)
        denominator = self.Normalization_function(all_label_combos, splitted)

        if denominator == 0:
            return 0

        return numerator / denominator


    def p_theta_numerator(self, sentence, labels):
        sum = 0.0

        for i in range(self.n_feature_functions):
            sum += self.weights[i] * self.apply_feature_function(sentence, labels, i)
        
        return np.exp(sum)


    def apply_feature_function(self, sentence, labels, feature_function_index):
        sum = 0.0
        
        for i in range(len(labels)):
            if i > 0:
                sum += self.funcdict[feature_function_index](sentence, labels[i], labels[i - 1], i)
            else:
                sum += self.funcdict[feature_function_index](sentence, labels[i], labels[i], i) # tf is the index error bug here???
        
        return sum

    def Normalization_function(self, all_labels, sentence): # all_labels should be 2d array of all different label combinations
        sum = 0.0

        for i in range(len(all_labels)):
            sum += self.p_theta_numerator(sentence, all_labels[i])

        return sum

    def all_label_combinations(self, n_labels):
        return np.array(list(itertools.combinations(self.all_labels, n_labels)))



    # OPTIMIZER FUNCTIONS (FOR TRAINING)
    # Should take a file, and output a set of sentences and corresponding labels
    # [[word1, word2, word3,...], sentence2, sentence3,...]
    # [[ label1, label2, label3 ], labels2, labels3,...]
    def read_txt(self, file_name): 
        sentences = []
        sentence_labels = []

        with open(file_name, "r", encoding='ISO-8859-1') as a_file:
            for line in a_file:
                stripped_line = line.strip()

                if len(stripped_line) != 0:
                    import re
                    res = re.split(r'\[|\]', stripped_line)

                    labels, sentence = self.read_sentence(res)
                    sentences.append(sentence)
                    sentence_labels.append(labels)

        return sentences, sentence_labels

    def read_sentence(self, sentence):
        labels = []
        sentence_clean = []

        def is_class(word):
            return self.all_labels.__contains__(word)
        
        def is_class_end(word):
            return len(word) > 2 and word[0] == '/' and self.all_labels.__contains__(word[1:])

        for i in sentence:
            if is_class(i):
                self.stack.append(i)
            elif is_class_end(i):
                self.stack.remove(i[1:])
            else:
                sentence_clean.append(i)
                labels.append(self.stack[-1])

        return labels, sentence_clean
        


    def train(self, all_sentences, all_labels, is_saved): # need to apply gradient descent algo here: w = w + alpha(all_feature_functions(x, y) - all_label_combos(all_feature_functions(x, y_i) * p_theta(y_i | x)))
        if is_saved:
            import pickle
            with open('trained_model.pickle', 'rb') as fp:
                self.weights = np.array(pickle.load(fp))
                fp.close()
            return

        # all_sentences = [] # set of all sentences for all files.
        # all_labels = [] # corresponding labels for all files' sentences

        # # # Get all training files
        # for filename in os.listdir(directory):
        #     f = os.path.join(directory, filename)
            
        #     if os.path.isfile(f):
        #         sentences, labels = self.read_txt(f)
        #         all_sentences.extend(sentences)
        #         all_labels.extend(labels)

        for i in range(self.n_feature_functions):
            for j in range(len(all_sentences)):
                self.weights[i] += self.alpha * self.gradient_descent(i, j, all_sentences, all_labels)
            
            print(self.weights)


        import pickle
        with open('trained_model.pickle', 'wb') as fp:
            pickle.dump(self.weights, fp)


    def gradient_descent(self, n_feature_func, j, all_sentences, all_labels):
        # todo, what sentence do we use and what corresponing label? Is this trained every single time we call p(y | x)?
        sentence = all_sentences[j]
        labels = all_labels[j]

        F_x_y = self.apply_feature_function(sentence, labels, n_feature_func)
        all_combos = self.all_label_combinations(len(sentence))
        total = 0.0

        for k in range(len(all_combos)):
            total += self.apply_feature_function(sentence, all_combos[k], n_feature_func) * self.p_theta(sentence, all_combos[k])
        
        return F_x_y - total


    # FEATURE FUNCTIONS
    # Returns true if word is fully uppercase, 0 otherwise
    def all_uppercase_function(self, sentence, i_word_label, i_prev_word_label, i):
        return (int) (sentence[i].isupper())
    
    # Returns true if word's first letter is uppercase, 0 otherwise
    def first_uppercase_function(self, sentence, i_word_label, i_prev_word_label, i):
        return (int) (sentence[0].isupper())

    def is_photo_function(self, sentence, i_word_label, i_prev_word_label, i):
        splitted = sentence[i].split()
        return (int) (splitted.__contains__('img') or i_word_label == 'IMG')

    # special word feature function
    def is_positive_word(self, sentence, i_word_label, i_prev_word_label, i):
        return (int) ( self.is_date(sentence, i_word_label, i_prev_word_label, i) or 
            self.is_phone_number(sentence, i_word_label, i_prev_word_label, i) or
            self.is_position(sentence, i_word_label, i_prev_word_label, i) or
            self.is_keyword(sentence, i_word_label, i_prev_word_label, i)
        )

    def is_english_word(self, sentence, i_word_label, i_prev_word_label, i):
        d = PyDictionary()
        return (int) (d.meaning(sentence[i]) != None)

    def is_date(self, sentence, i_word_label, i_prev_word_label, i):
        splitted = sentence[i].split()
        return (splitted.__contains__('Phd') or splitted.__contains__('Bs') or splitted.__contains__('Ms'))
    
    def is_phone_number(self, sentence, i_word_label, i_prev_word_label, i):
        return (i_word_label == 'photo' or i_prev_word_label == 'photo')
    
    def is_position(self, sentence, i_word_label, i_prev_word_label, i):
        splitted_space = sentence[i].split()
        return (i_word_label == 'position' or i_prev_word_label == 'position' or splitted_space.__contains__('professor'))

    def is_keyword(self, sentence, i_word_label, i_prev_word_label, i):
        splitted_space = sentence[i].split()
        flag = False

        for i in splitted_space:
            if self.all_labels.__contains__(i):
                flag = True

        return flag or self.all_labels.__contains__(i_word_label)



    # Testing
    def test_sentence(self, sentence, actual_labels):
        sentence = sentence.split()
        all_combos = self.all_label_combinations(len(sentence))
        max_c = -1
        index = 0

        for i in range(len(all_combos)):
            prob = self.p_theta(sentence, all_combos[i])
            
            if max_c < prob:
                max_c = prob
                index = i

        return all_combos[index]

    # given a file, output all file sentences and appropriate labels
    def find_page_labels(self, directory, filename):
        f = os.path.join(directory, filename)
            
        if os.path.isfile(f):
            sentences, labels = self.read_txt(f)
            
            for i in range(len(sentences)):
                all_combos = self.all_label_combinations(len(sentences[i]))
                max_c = 0.0
                index = -1
                
                for j in all_combos:
                    
                    if max_c < self.p_theta(sentences[i], j):
                        index = j
                
                for j in sentences[index]:
                    if self.all_labels.__contains__(j):
                        print(self.all_labels.index(j) + ': ' + sentences[i])



    # returns train_sentences, train_labels, test_sentences, test_labels as LISTS. Is the splitted result of the 
    # current dataset 
    def split_into_folds(self, all_sentences, all_labels, folds, num_test_fold, num_folds): 
        s = np.array(all_sentences, dtype=object)
        l = np.array(all_labels, dtype=object)
        n = len(all_sentences)
        
        chunk_size = int(n / num_folds)
        start_test = chunk_size * num_test_fold
        end_test = chunk_size * (num_test_fold + 1)

        train_sentences = np.append(s[0:start_test], s[end_test:n])
        train_labels = np.append(l[0:start_test], l[end_test:n])
        test_sentences = s[start_test:end_test]
        test_labels = l[start_test:end_test]

        return train_sentences.tolist(), train_labels.tolist(), test_sentences.tolist(), test_labels.tolist()

    def KFOLD_cross_validation(self, all_sentences, all_lables, folds):
        total = 0.0
        correct = 0.0
        grand_total = 0.0
        grand_correct = 0.0

        # this is our precision recall matrix. The rows represent the actual outputs, and the columns represent the predicted ones        
        pr_matrix = np.zeros((self.n_feature_functions, self.n_feature_functions))

        for i in range(folds):
            train_sentences, train_labels, test_sentences, test_labels = self.split_into_folds(all_sentences, all_labels, folds, i, folds)
            self.train(train_sentences, train_labels, False)

            # checking the predicted values of the classifier
            for j in range(len(test_labels)):
                res = self.p_theta(test_sentences[j], test_labels[j])
                
                if res == test_labels[j]:
                    correct += 1
                
                for k in range(len(res)):
                    if res[k] == test_labels[j][k]:
                        index_pred = self.all_labels.index(res[k]) # represents index of the predicted label
                        index_actual = self.all_labels.index(test_labels[j][k])

                        pr_matrix[index_actual][index_pred] += 1 # incrementes the correct value in the matrix



            total += len(test_labels)
            print('fold' + i + ' accuracy:' + correct / total)
            
            grand_total += total
            grand_correct += correct

            total = 0.0
            correct = 0.0
        
        print('grand total accuracy:' + grand_correct / grand_total)
        
        precision, recall = self.get_pr(pr_matrix)

        print('precision: ' + precision)
        print('recall: ' + recall)

    def get_pr(self, pr_matrix):
        precision = 0.0
        recall = 0.0

        for i in range(len(pr_matrix)):
            horizontal = np.sum(pr_matrix, axis=0)
            vertical = np.sum(pr_matrix, axis=1)

            precision += pr_matrix[i][i] / horizontal
            recall += pr_matrix[i][i] / vertical
            

        return precision, recall

    def page_tester(self, directory):
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            
            if os.path.isfile(f):
                sentences, labels = self.read_txt(f)
                labels = np.array(labels)
                print(labels)
                return sentences, labels.tolist()

    def build_profile(self, webpagename):
        sentences, labels = self.read_txt(webpagename) # sentences should have values, labels should all just say standard word

        for i in sentences:
            all_label_combos = self.all_label_combinations(len(i))
            probs = []

            for k in range(len(all_label_combos)):
                probs.append(self.p_theta(i, all_label_combos[k]))
            
            predicted_labels = all_label_combos[probs.index(max(probs))] # this is our predicted set of labels for the particular sentence
            
            for l in range(len(predicted_labels)):
                if self.all_labels.__contains__(predicted_labels[l]) and predicted_labels[l] != 'standard word':
                    print(predicted_labels[l] + ': ' + i[l]) # prints out the resulting label for a certain webpage


# DBLP
# Create a publication crawler to retrieve researcher information from a publication website (e.g.: DBLP). Returns a list of publication urls.
def crawl_DBLP(researcher_name):
    dblp_url = 'https://dblp.org/search?q=' + researcher_name
    soup = BeautifulSoup(requests.get(dblp_url).text, 'html.parser')

    for link in soup.find_all('a'):
        print(link.get('href'))


# This is your main entry point. Uncomment what you wish to test/try
if __name__ == "__main__":
    # just a sample sentence
    # sentence = "I am Bob"

    # instiantiating the model
    # CRF_MODEL = feature_functions()

    # all_sentences = [] # set of all sentences for all files.
    # all_labels = [] # corresponding labels for all files' sentences

    # for testing with a singular trained page
    # some_sentences, some_labels = CRF_MODEL.page_tester('898_data')
    # OR Get all training files
    # for filename in os.listdir('898_data'):
    #     f = os.path.join('898_data', filename)
        
    #     if os.path.isfile(f):
    #         sentences, labels = CRF_MODEL.read_txt(f)
    #         all_sentences.extend(sentences)
            # all_labels.extend(labels)


    # CRF_MODEL.KFOLD_cross_validation(all_sentences, all_labels, 10)


    # trains model
    # CRF_MODEL.train(all_sentences, all_labels, True)
    
    # exists to test a new webpage with no labels
    # CRF_MODEL.build_profile('unlabeled_webpages/Bernstein.txt')

    # exists to test a singular sentence against a singular set of labels
    # print(CRF_MODEL.test_sentence(sentence, ["standard word", "standard word", "name"]))
