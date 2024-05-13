# question-answer-bot
This repository contains the code for a Question-Answer (QA) bot using a deep learning approach to answer your queries on Artificial Intelligence. 


Data set

The dataset consists of text from one or more Wikipedia pages on Artificial Intelligence. The text data from these HTML Wikipedia pages are scraped using the BeautifulSoup library for efficient parsing of text content. The text data is stored in a csv file under the 'answer' column.

Question Creator

Manually creating questions for enormous text data is a time-consuming process. So, in this project, the process of creating questions based on the text under the 'answer' column has been automated using a question creator script, which takes as input a csv file with the 'answer' column. I implement a keyword extractor script.

The algorithm for the keyword extractor is as follows:
1. Read csv containing the 'answer' column and store it in a Pandas dataframe
2. Use the RAKE (Rapid Automatic Keyword Extraction) library from rake_nltk to extract keywords from text
3. Rank the keywords based on the frequency of individual words and the degree of word co-occurrence within the text. Phrases with higher scores reflect higher relevance in the context of the document.
4. Select the top 20 keywords from each answer column to generate questions in such a way that only the multi-word phrases are considered during question creation
5. Create a question template using the most common question verbs and append this question template to the extracted keywords to get valid questions.
6. Create a new column in the existing 

Data Preprocessing

The question-answer data is stored in a csv file called 'data.csv'. To make the text data suitable to be given as input to a deep learning model and also for better information processing, certain preprocessing steps are implemented on the text and are as follows:
1. All the text is converted to lower case letters to ensure uniformity in text processing
2. Non-alphanumeric characters and punctuation are removed from the text using regex functions
3. The text data is tokenized into words using word_tokenize library from nltk
4. Stopwords are removed from the text

Data split

The dataset is split into train and test sets following a 80:20 ratio for train:test respectively.

Creating the vocabulary

Unique tokens are taken from the text and a mapping is created from token: index.
Special tokens are added:
1. vocab[<pad>] for padding sequences to the same length
2. vocab[<start>] to initiate the sequence generation
3. vocab['<end>'] to end the sequence generation
4. token-to-ińdex conversion is done to the question tokens and answer tokens columns
5. The max index used in the question and answer is used to set the vocab size

Model Training

A sequence-to-sequence model using LSTM is used for generating the answers based on user query. The model architecture consists of an encoder and a decoder to generate answers.
The hyperparameters used for the model are as follows:
embedding_dim = 128
hidden_dim = 256
learning_rate = 0.001
num_epochs = 100
The batch size was altered to get the desired output. 
Further experiments are needed to set the hyperparameters to the best possible ones by using SWEEP functionality provided by Weights and Biases. Due to time constraints of the project, this is considered as future work.
Pytorch is used for creating and saving the model.
The loss function used for this task is the Cross Entropy loss and Adam optimizer is used to optimize the loss function.
The model was trained for 100 epochs on an NVIDIA RTX 3070 GPU for around 60 minutes.

QA webservice

Flask is used to deploy the QA bot model as a web service.
The user has to enter the desired question in the box provided to get an answer from the QA bot
The maxuímum length of the generated answer is set to 50





