### Author: Prasanth Murali ###

### Import Modules Here ###
from __future__ import division
print "...importing modules..."
import string
import re
from collections import Counter
import numpy
import matplotlib.pyplot as plt
import os

### References: ###
## 1. https://github.com/HackHarassment/TwitterClassifier
## 2. http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
## 3. https://codereview.stackexchange.com/questions/109632/latent-dirichlet-allocation-in-python
## 4. http://www.indjst.org/index.php/indjst/article/view/93825
## 5. http://stackoverflow.com/questions/184618/what-is-the-best-comment-in-source-code-you-have-ever-encountered 

### Constants: ###
lambda_value = 0.1
prior_class_probability_good = 0.6
prior_class_probability_bad = 0.4
alpha = 100
beta = 5
iters = 1000
max_epochs = 100

### Function Definitions ###

'''
String -> ListofWords
GIVEN: A filename
RETURNS: The list of words from that file, removing the punctions
         and in lower case
'''         
def analyzer(filename):
    tokens=[]
    i=0
    review=open(filename,'r')
    with open(filename) as f:
        for line in f:
            if i < 10000:
                words = [x.strip(string.punctuation) for x in line.lower().split()]
                tokens=tokens+words
                i=i+1
            else:
                break      
    return tokens

'''
ListofWords, String -> ListofNumber
GIVEN: A Vocabulary and filename
RETURNS: Quality of occurrence of the vocabulary in that file
'''
def quality(vocabulary, corpus_filename):
    with open(corpus_filename) as corpus:
        return [vocabulary.index(word) for line in corpus for word in line.split()]

'''
ListOfWords ListofWords -> Integer
GIVEN: Two list of words
RETURNS: The Counter and total vocabulary count from the two list of words
'''
def get_vocab_size(bad_tokens,good_tokens,test_tokens):
    overall_vocab=bad_tokens+good_tokens+test_tokens
    count = {}
    count = Counter(overall_vocab)
    return count,len(count)

'''
GIVEN: No arguments
RETURNS: The dictionary of test tweets
'''
def get_test_data():
    with open("Test_Data.txt") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    f=open("test_data_1.txt",'a')
    import string
    dictionary_of_test_data={}
    i=0
    for line in content:

        words=[x.strip(string.punctuation) for x in line.lower().split()]
        dictionary_of_test_data["Tweet"+str(i)]=words
        i=i+1
        for word in words:
            f.write(word+" ")
        f.write("\n")    
    f.close()
    return dictionary_of_test_data

'''
ListofNumber ListofTopics Number -> Values for the Gibbs Sampling
GIVEN: The values, ListofTopics and length of values
RETURNS: The starting values for LDA (Gibbs Sampling)
'''
def generate_starting_LDA_state(values, M, V):
    I = len(values)
    Na = numpy.zeros((I, M)) + alpha     
    Nb = numpy.zeros(I) + M*alpha       
    Nc = numpy.zeros((M, V)) + beta     
    Nd = numpy.zeros(M) + V*beta

    '''
    Function to Initialize LDA values
    '''
    def helper(i, w):
        m = numpy.random.randint(0, M)
        Na[i, m] += 1
        Nb[i] += 1
        Nc[m, w-1] += 1
        Nd[m] += 1
        return m

    return Na, Nb, Nc, Nd, [[helper(i, w) for w in value] for i, value in enumerate(values)]

'''
Dictionary -> Dictionary
GIVEN: The test dictionary
RETURNS: Performs Naive_Bayes to the test data and returns the classification
         of test data as belonging to Abusive or Non Abusive Tweets in
         a dictionary of probabilities corresponding to that tweet
         good - corresponds to non abusive tweets
         bad - corresponds to abusive tweets
FORMULA: P(C|X) = P(X|C) * P(C) / P(X)
'''
def naive_bayes(dictionary_of_test_data):
    classification={}
    for test in dictionary_of_test_data.keys():
        tweet=dictionary_of_test_data[test]
        good_prob = get_probability(tweet,0)
        bad_prob= get_probability(tweet,1)
        classification[test]=[good_prob,bad_prob]
    return classification

'''
ListofWords ListofString -> ListofFloat ListofWords ListofWords
GIVEN: The ListofTopics and The list of filenames
RETURNS: The distance values after performing LDA, The vocabulary
         and the words associated with each topic
'''
def LDA(topics, *filenames):

    ## analyzing the files to get Counter of words from them ##
    words = Counter()
    for corpus_file in filenames:
        with open(corpus_file) as corpus:
            words.update(word for line in corpus for word in line.split())                 
            
    vocabulary = words.keys()
    values = [quality(vocabulary, corpus) for corpus in filenames]
    Na, Nb, Nc, Nd, topic_of_words_per_value = generate_starting_LDA_state(values, topics, len(vocabulary))    

    ## Gibbs Sampling ##
    probabilities = numpy.zeros(topics)
    for _ in xrange(iters):

        for i, value in enumerate(values):
            topic_per_word = topic_of_words_per_value[i]
            for n, w in enumerate(value): 
                m = topic_per_word[n]      

                Na[i, m] -= 1
                Nb[i] -= 1
                Nc[m, w-1] -= 1
                Nd[m] -= 1

                ## computing topic probability ##
                probabilities[m] = Na[i, m] * Nc[m, w-1]/(Nb[i] * Nd[m])
                
                ## choosing new topic based on this ##
                q = numpy.random.multinomial(1, probabilities/probabilities.sum()).argmax()
                
                ## assigning word to topic ##
                topic_per_word[n] = q

                Na[i, q] += 1
                Nb[i] += 1
                Nc[q, w-1] += 1
                Nd[q] += 1

    distances = Nc/Nd[:, numpy.newaxis]
    return distances, vocabulary, words

'''
Dictionary -> File
GIVEN: The dictionary containing probability values
       of tweets after Naive Bayes and the threshold for classification
RETURNS: The results of tweets (as good or bad) onto
         Naive_Bayes_Model_Results.txt and the same as
         a dictionary
'''
def get_NB_results(NB_Classification, threshold):
    f = open("Naive_Bayes_Model_Results.txt",'w')
    answer_dict={}
    for key in NB_classification.keys():
        value = NB_classification[key]
        ratio = value[0]/value[1]
        if ratio < threshold:
            answer_dict[key]="Good"
            f.write(key+" "+"Good"+"\n")
        else:
            answer_dict[key]="Bad"
            f.write(key+" "+"Bad"+"\n")
    f.close()
    return answer_dict

'''
Dictionary ListofWords -> Float Float Float Float
GIVEN: The model output and expected output
RETURNS: The number of true positives, true negatives,
         false positives, false negatives
'''         
def get_PR(NB_results,expected_output):
    tp=0
    tn=0
    fp=0
    fn=0
    for key in NB_results.keys():
        model_outcome = NB_results[key]
        expected_output_index = int(key.split('et')[1])
        expected_outcome = expected_output[expected_output_index]
        if model_outcome == 'Bad' and expected_outcome == 'Bad':
            tp = tp + 1
        elif model_outcome == 'Good' and expected_outcome == 'Bad':
            fn = fn + 1
        elif model_outcome == 'Bad' and expected_outcome == 'Good':
            fp = fp + 1
        else:
            tn = tn + 1
    return tp,tn,fp,fn                      

'''
GIVEN: No arguments
RESULTS: reads the expected output from Test_Answers.txt
         with line number corresponding to tweet number
         and the string corresponding to its classified
         class
'''
def read_actual_answers():
    f=open("Test_Answers.txt",'r')
    lines = f.readlines()
    answers=[]
    for i in range(len(lines)):
        answers=answers+[lines[i].split('\n')[0]]
    return answers
        
'''
ListofWords Integer -> Float
GIVEN: A tweet as a list of words and a number to denote the probability
       class (abusive or not) that we are trying to calculate
       num = 0 means non abusive tweets
       num = 1 means abusive tweets
RETURNS: The probability of tweet as belonging to that particular class
'''
def get_probability(tweet,num):
    prob=1
    if num==0:        
        for word in tweet:
            if word in prob_good_words.keys():
                term1=prob_good_words[word]
                term2=prior_class_probability_good
                term3=vocab_counter[word]/vocab_size
                prob = prob * (term1 * term2) / term3
            else:
                term1=1/len(prob_good_words)
                term2=prior_class_probability_good
                term3=vocab_counter[word]/vocab_size
                prob = prob * (term1 * term2) / term3
        return prob            
    else:
        for word in tweet:
            if word in prob_bad_words.keys():                
                term1=prob_bad_words[word]
                term2=prior_class_probability_bad
                term3=vocab_counter[word]/vocab_size
                prob = prob * (term1 * term2) / term3
            else:
                term1=1/len(prob_bad_words)
                term2=prior_class_probability_bad
                term3=vocab_counter[word]/vocab_size
                prob = prob * (term1 * term2) / term3
        return prob

'''
ListofNumbers ListofNumbers -> Number
GIVEN: Two List of numbers
RETURNS: The sum squared difference between corresponding values
         of the two lists
'''
def get_sumsq_error(list1,list2):
    sum_error=0
    for i in range(0,len(list1)):
        diff=float(list1[i])-float(list2[i])
        sum_error=sum_error+diff*diff
    return sum_error          

'''
Dictionary Integer -> Dictionary
GIVEN: a dictionary with (words & their counts) in a particular class
       and total number of words in that particular class
RETURNS: A probabilistic dictionary of those words
''' 
def get_prob_dict(tokens_count, length):
    count={}
    for key in tokens_count.keys():
        count[key]=(tokens_count[key]+lambda_value)/(length + lambda_value * vocab_size)
    return count

### main() portion of the program ###
print "...starting main..."
##
##Analyse the bad_corpus.txt, getting all the word counts
##and write them on to bad_counts.txt
##
bad_tokens=analyzer('bad_corpus.txt')
total_bad_words = len(bad_tokens)
bad_tokens_count = Counter(bad_tokens)
with open('bad_counts.txt','w')as g:
    for key in bad_tokens_count.keys():
        g.write(key+" "+str(bad_tokens_count[key]))
        g.write("\n")

##
##Analyse the good_corpus.txt, getting all the word counts
##and write them on to good_counts.txt
##
good_tokens=analyzer('good_corpus.txt')
total_good_words = len(good_tokens)
good_tokens_count = Counter(good_tokens)
with open('good_counts.txt','w')as g:
    for key in good_tokens_count.keys():
        g.write(key+" "+str(good_tokens_count[key]))
        g.write("\n")

## get test data in the form of a dictionary of tweets ##
dictionary_of_test_data=get_test_data()         

##
##Analyse the test_data.txt, getting all the word counts
##and write them on to test_counts.txt
##
test_tokens=analyzer("Test_Data.txt")
total_test_words = len(test_tokens)
test_tokens_count = Counter(test_tokens)
with open('test_counts.txt','w')as g:
    for key in test_tokens_count.keys():
        g.write(key+" "+str(test_tokens_count[key]))
        g.write("\n")        

## get the total vocabulary size here ##
vocab_counter,vocab_size = get_vocab_size(bad_tokens,good_tokens,test_tokens) 

## Get the probability counts ##
prob_bad_words=get_prob_dict(bad_tokens_count,total_bad_words)
prob_good_words=get_prob_dict(good_tokens_count,total_good_words)

## Write bad_prob_counts onto bad_counts_prob.txt ##
with open('bad_counts_prob.txt','w')as g:
    for key in prob_bad_words.keys():
        g.write(key+" "+str(prob_bad_words[key]))
        g.write("\n")
        
## Write good_prob_counts onto good_counts_prob.txt ##
with open('good_counts_prob.txt','w')as g:
    for key in prob_good_words.keys():
        g.write(key+" "+str(prob_good_words[key]))
        g.write("\n")

## get test data in the form of a dictionary of tweets ##
dictionary_of_test_data=get_test_data()        

## apply naive bayes here ##
print "...applying Naive Bayes..."
NB_classification = naive_bayes(dictionary_of_test_data)

pr_dict={}
threshold=0
while threshold<max_epochs:
    # print "...writing NB results onto a file..."
    NB_results=get_NB_results(NB_classification,threshold)

    # print "...reading actual test results..."
    expected_output_list=read_actual_answers()

    ## PRECISION - RECALL values for naive bayes ##
    true_positives,true_negatives,false_positives,false_negatives = get_PR(NB_results,expected_output_list)
    if true_positives == 0:
        precision=0
        recall=0
    else:            
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        pr_dict[str(threshold)]=[precision,recall]
        threshold = threshold + 0.01
    
## Plotting PR Curve ##
print "...starting PR curve analysis..."
recall_values=[0.1]*10000
precision_values=[0.1]*10000
f1_values=[0.1]*10000
i=0
while i<max_epochs:
    a=int(i*100) ## so that list indices are integers
    recall_values[a]=pr_dict[str(i)][1]
    precision_values[a]=pr_dict[str(i)][0]
    f1_values[a]=2*recall_values[a]*precision_values[a]/(recall_values[a]+precision_values[a])
    i=i+0.01

## "...Writing f1 values onto Threshold_F1.txt..." ##
f=open("Threshold_F1.txt",'w')
for f1_value in f1_values:
    f.write(str(f1_value))
    f.write("\n")
f.close()    
## "...Finished Writing f1 values onto Threshold_F1.txt..." ##

print "...starting to plot...saving to PR Curve.jpg"   
fig = plt.figure()
plt.plot(recall_values,precision_values)
fig.suptitle('Precision Recall Curve', fontsize=20)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=16)
fig.savefig('PR Curve.jpg')
print "...plotting finished..."

## Starting LDA Analysis here ##
print "...starting LDA(latent dirichlet allocation)..."
## initializing topic count ##
topics = 20

## LDA for the bad_corpus.txt - abusive tweets ##
print "...generating LDA for bad file..."
distances, vocabulary, words_count = LDA(topics, 'bad_corpus.txt')
f=open('bad_LDA.txt','w')
for topic in range(0,topics):
    distance=0
    for word_index in numpy.argsort(-distances[topic])[:20]:
        word = vocabulary[word_index]
        distance = distance + distances[topic, word_index]
    f.write(str(distance))
    f.write("\n")
print "done with bad file"
f.close()

## LDA for the good_corpus.txt - non abusive tweets ##
print "...generating LDA for good file..."
distances, vocabulary, words_count = LDA(topics, 'good_corpus.txt')
f=open('good_LDA.txt','w')
for topic in range(0,topics):
    distance=0
    for word_index in numpy.argsort(-distances[topic])[:20]:
        word = vocabulary[word_index]
        distance = distance + distances[topic, word_index]
    f.write(str(distance))
    f.write("\n")
print "done with good file"
f.close()

## LDA for bad test file ##
print "...generating LDA for bad test file..."
distances, vocabulary, words_count = LDA(topics, 'Test_LDA_Bad.txt')
f=open('Test_bad_LDA.txt','a')
for topic in range(0,topics):
    distance=0
    for word_index in numpy.argsort(-distances[topic])[:20]:
        word = vocabulary[word_index]
        distance = distance + distances[topic, word_index]
    f.write(str(distance))
    f.write("\n")        
print "done with test bad file"
f.close()

## LDA for good test file ##
print "...generating LDA for good test file..."
distances, vocabulary, words_count = LDA(topics, 'Test_LDA_good.txt')
f=open('Test_good_LDA.txt','a')
for topic in range(0,topics):
    distance=0
    for word_index in numpy.argsort(-distances[topic])[:20]:
        word = vocabulary[word_index]
        distance = distance + distances[topic, word_index]
    f.write(str(distance))
    f.write("\n")        
print "done with test good file"
f.close()

print "...LDA complete..."

## Getting the values from file onto a list for computation ##
## Bad file, Good file, Bad test file, Good test file in that order ##

f = open('bad_LDA.txt','r')
lines = f.readlines()
bad_values=[]
for word in lines:
    bad_values = bad_values + [word.split('\n')[0]]
f.close()

f = open('good_LDA.txt','r')
lines = f.readlines()
good_values=[]
for word in lines:
    good_values = good_values + [word.split('\n')[0]]
f.close()

f = open('Test_bad_LDA.txt','r')
lines = f.readlines()
test_bad_LDA_values=[]
for word in lines:
    test_bad_LDA_values = test_bad_LDA_values + [word.split('\n')[0]]
f.close()

f = open('Test_good_LDA.txt','r')
lines = f.readlines()
test_good_LDA_values=[]
for word in lines:
    test_good_LDA_values = test_good_LDA_values + [word.split('\n')[0]]
f.close()

## Calculate Sum Sq errors here for both the good and bad test files ##
## with both the good and the bad corpora ##

squared_error_bad_file_with_bad_corpus = get_sumsq_error(bad_values,test_bad_LDA_values)
squared_error_bad_file_with_good_corpus = get_sumsq_error(good_values,test_bad_LDA_values)

f = open("LDA_results.txt",'w')

f.write("For the bad test file, the error with bad corpus is less than error with good corpus, as seen below \n")
f.write("Error with Bad corpus is:"+str(squared_error_bad_file_with_bad_corpus)+"\n")
f.write("Error with Good corpus is:"+str(squared_error_bad_file_with_good_corpus)+"\n")
f.write("_______________________________________________________________________________________________________\n")

print "For the bad test file, the error with bad corpus is less than error with good corpus, as seen below"
print "Error with Bad corpus is:"
print squared_error_bad_file_with_bad_corpus
print "Error with Good corpus is:"
print squared_error_bad_file_with_good_corpus

squared_error_good_file_with_bad_corpus = get_sumsq_error(bad_values,test_good_LDA_values)
squared_error_good_file_with_good_corpus = get_sumsq_error(good_values,test_good_LDA_values)

f.write("For the good test file, the error with good corpus is less than error with bad corpus, as seen below \n")
f.write("Error with Bad corpus is:"+str(squared_error_good_file_with_bad_corpus)+"\n")
f.write("Error with Good corpus is:"+str(squared_error_good_file_with_good_corpus)+"\n")
f.write("_______________________________________________________________________________________________________")

print "For the good test file, the error with good corpus is less than error with bad corpus, as seen below"
print "Error with good corpus is:"
print squared_error_good_file_with_good_corpus
print "Error with bad corpus is:"
print squared_error_good_file_with_bad_corpus

f.close()

## Ending the project on a light note ##
print "For the brave souls who have waited this long: You are the chosen ones,"
print "the valiant knights of programming who toil away, without rest,"
print "awaiting our most awful code. To you, true saviors, kings of men,"
print "I say this: never gonna give you up, never gonna let you down,"
print "never gonna run around and desert you. Never gonna make you cry,"
print "never gonna say goodbye. Never gonna tell a lie and hurt you."
print "Program finishes gracefully..THE END"
