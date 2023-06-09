# Importing Libraries

''' General approach (methods of data pre-processing specifically) referenced from 
https://pykit.org/chatbot-in-python-using-nlp/ and
https://www.projectpro.io/article/python-chatbot-project-learn-to-build-a-chatbot-from-scratch/429;
the code itself was not referenced, just the necessary order of
tokenizing, lemmatizing, etc. Everything else that isn't marked
was just prior knowledge from learning in the spring or was figured out
through toying and testing in a different CoLab notebook. '''

import nltk
nltk.download('punkt')                                                   
nltk.download('wordnet')                                                  
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import keras.preprocessing
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime 
# %load_ext tensorboard

# Importing and Designating the Datafile
datafile = open("asepData.JSON").read()
data = json.loads(datafile)
eval_datafile = open("evalData.JSON").read()
evalData = json.loads(eval_datafile)

# Importing and Tokenizing Data

words = [] # Where tokenized words from data will go
classes = [] # Set of classes/categories (will add to as datafile is finished)
x_set = []
y_set = []
# Exclude punctuation from evaluation of class of input
ignore_punc = ['?', '!', ',', '.', '-', "'"]
''' Code for parsing the datafile; referenced from bottom section of 
realpython.com/python-json/. I am pretty unfamiliar with JSON, but it's a 
clean and easy way for me to store and access data, 
so I'm trying it out. '''
for intent in data['intents']: # looking at each category/set
  for question in intent['questions']: # looking at each question
    new_vocab = nltk.word_tokenize(question) # tokenizing the question
    words.extend(new_vocab) # adding the words to the words list
    x_set.append(question) # adding to x dataset
    y_set.append(intent['class']) # adding to y dataset

    if intent['class'] not in classes:
      classes.append(intent['class']) # getting the list of classes

# Lemmatizating Data

'''Did not reference code on here, but did read 
www.nltk.org/api/nltk.stem.wordnet.html#module-nltk.stem.wordnet 
to understand how the function and process work.'''
lemmatizer = WordNetLemmatizer() # Creating a lemmatizer object
tempWords = []

for index, word in enumerate(words): # cycling through each word in the list
  # NTLK's tokenize doesn't exclude punctuation, must be done manually
  if word not in ignore_punc: 
    tempWords.append(lemmatizer.lemmatize(word.lower())) 
    # lemmatizing and making lowercase before adding to list (for uniformity)

words = list(set(tempWords)) # removing duplicates
# this is the "dictionary" that the BOW will be based on

# Preparing Training Data

# it's the class output list/array that corresponds to the BOW
full_X = []
full_Y = []

for index, question in enumerate(x_set): # going through x_set
# also, using enumerate() makes it easy to keep track of the active index for identifying the y-value
  bag_of_words = []
  cur_class = y_set[index] # denoting which class the question is in
  question = lemmatizer.lemmatize(question.lower())

  for word in words: # creates a bag of word model (identifies if a word is present in the question)
  # works since it feeds the model what words are present along with the overall assigned class
    if word in question:
      bag_of_words.append(1)
    else:
      bag_of_words.append(0)

  output_temp = [0] * len(classes) # duplicating the empty set
  output_temp[classes.index(cur_class)] = 1 # setting the corresponding class as 1

  full_X.append(bag_of_words)
  full_Y.append(output_temp)
  # training.append([bag_of_words, output_temp]) # adding the X, Y pair to the training set

training_X = np.array(full_X) # converting to numpy array
training_Y = np.array(full_Y)

# Evaluation data prep; very, very similar to training data prep, just diff datafile
eval_X = []
eval_Y = []
for intent in evalData['intents']: # looking at each category/set
  for question in intent['questions']: # looking at each question
    eval_X.append(question) # adding to x dataset
    eval_Y.append(intent['class']) # adding to y dataset
final_X = []
final_Y = []
for index, question in enumerate(eval_X): # going through x_set
# also, using enumerate() makes it easy to keep track of the active index for identifying the y value
  bag_of_words = []
  cur_class = eval_Y[index] # denoting which class the question is in
  question = lemmatizer.lemmatize(question.lower())

  for word in words: # creates a bag of word model (basically identifies if a word is present in the question or not)
  # it works since it feeds the model what words are present and how that relates to the assigned class
    if word in question:
      bag_of_words.append(1)
    else:
      bag_of_words.append(0)

  output_temp = [0] * len(classes) # duplicating the empty set
  output_temp[classes.index(cur_class)] = 1 # setting the corresponding class as 1

  final_X.append(bag_of_words) # setting up X set
  final_Y.append(output_temp) # and the Y set

final_X = np.array(final_X) # converting to numpy array
final_Y = np.array(final_Y)

# Embeddings Layer Prep
'''
# Either this or the other data preparation sections run -- they are not to be ran simultaneously,
# as BOW and embeddings are different
# I don't recommend trying the embeddings (they're really bad with the small
# dataset), but since I did write this code, it is listed here for reference.
import keras.preprocessing
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Choosing how to represent out-of-vocabulary words
tokenizer = Tokenizer(oov_token = "<OOV>")
x_set = []
for intent in data['intents']: # looking at each category/set
  for question in intent['questions']: # looking at each question
    x_set.append(question) # adding to x dataset
# Fitting the vocabulary to the words in each training question
tokenizer.fit_on_texts(x_set)
print(x_set)
print(tokenizer.word_index)
# Converting the questions to sequences & padding them (so length is consistent).
# Truncating is, by default, 'pre' instead of 'post' and in my opinion that's preferable
# since the first few words of most questions are interrogatives ("what", "where")
# and don't give much information on the category of the question.
sequences = tokenizer.texts_to_sequences(x_set)
padded_x = pad_sequences(sequences, padding = 'post', maxlen = 15)
print(padded_x)
# Setting that to the training set
training_X = padded_x

# Evaluation set
eval_X = []
for intent in evalData['intents']: # looking at each category/set
  for question in intent['questions']: # looking at each question
    eval_X.append(question) # adding to x dataset
# Pretty much same process as before
eval_sequences = tokenizer.texts_to_sequences(eval_X)
padded_eval = pad_sequences(eval_sequences, padding = 'post', maxlen = 15)
final_X = padded_eval
'''

# Model Design

# This current one here is actually the optimal model! (gives lowest loss & highest accuracy)
# Surprisingly simple. The more complex ones (including dropout layers, embeddings, etc.)
# have really high loss + don't do well.
# I think this could be because the dataset is incredibly small and specific.
model = keras.Sequential([
    # Just keeping the embeddings code as more context; replace the first dense layer
    # with this code if wanting to test the embeddings
    # keras.layers.Embedding(len(tokenizer.word_index) + 1, 64, input_length = 15),
    keras.layers.Dense(16, input_shape = (len(training_X[0]),), activation = 'relu'),
    keras.layers.Dense(len(training_Y[0]), activation = 'softmax')
])

# Compiling (these were also tested, and Adam w/ this parameter works best)
opt = tf.keras.optimizers.Adam(beta_1 = 0.7)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

# model.summary() # to view trainable parameters + details of the model structure

# Training Process and Statistics

''' Referenced https://www.tensorflow.org/tensorboard/scalars_and_keras to
learn how to use TensorBoard with TensorFlow, so some code comes from there.'''
# logdir = "ASEPlogs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir) 

# tf.debugging.set_log_device_placement(True) # checking if  GPU is used for training
# commented out bc Google CoLab limited my GPU usage (and my laptop is not CUDA-capable)

model.fit(training_X, training_Y, verbose = 0, epochs = 5000, callbacks=[tensorboard_callback])
# standard training; no training logs needed since tensorboard is running

# displays all runs and their statistics 
# (this part was done on CoLab and doesn't work here on GitHub)
# %tensorboard --logdir ASEPlogs/scalars/

# Model Evaluation
model.evaluate(final_X, final_Y)

# User Input

# Cleaning/Preparing User Input
def clean_up(user_input):
  # Tokenize
  user_tokens = nltk.word_tokenize(user_input)
  # Lemmatize, lowercase
  user_final = []
  for word in user_tokens:
    user_final.append(lemmatizer.lemmatize(word.lower()))
  # Create the BOW model
  bag_of_words = [0] * len(words)
  for index, word in enumerate(words):
      if word in user_final:
        bag_of_words[index] = 1
  # Returning fully setup BOW
  return np.array(bag_of_words)

# Model Prediction
def prediction(bow):
  # Feed BOW to model
  return model.predict(np.array([bow]))[0]
  # Returns list of class  probabilities

# Model Response
def get_response(probs):
  responses = []
  for intent in data['intents']:
    for response in intent['response']:
      responses.append(response) # collecting the list of responses
  maxProb = np.max(probs) # finding maximum probability
  index = (np.where(probs == maxProb)[0])[0] # identifying the corresponding class w/ probability
  return responses[index] # returning that category's response

# Starting Conversation

# Originally was going to implement a GUI with Tkinter, but I hated how it looked, so it's gone
def start_chat():
  print("Hello! Ask me any questions you may have about St. Paul's. If you're done asking questions, just say DONE.")
  while True:
    user = input("")
    if(user == 'DONE'):
      break;
    print(get_response(prediction(clean_up(user))))
  print("Hope I was able to help! Have a great day.")

start_chat()
