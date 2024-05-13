# question-answer-bot
This repository contains the code for a Question-Answer (QA) bot using a deep learning approach to answer your queries on Artificial Intelligence. 


**Data set**

The dataset consists of text from one or more Wikipedia pages on Artificial Intelligence. The text data from these HTML Wikipedia pages are scraped using the BeautifulSoup library for efficient parsing of text content. The text data is stored in a csv file under the 'answer' column. *Code: qa_data_pipeline.py*

**Question Creator**

Manually creating questions for enormous text data is a time-consuming process. So, in this project, the process of creating questions based on the text under the 'answer' column has been automated using a question creator script, which takes as input a csv file with the 'answer' column. I implement a keyword extractor script. *Code: question_creator.py*

The algorithm for the keyword extractor is as follows:
1. Read csv containing the 'answer' column and store it in a Pandas dataframe
2. Use the RAKE (Rapid Automatic Keyword Extraction) library from rake_nltk to extract keywords from text
3. Rank the keywords based on the frequency of individual words and the degree of word co-occurrence within the text. Phrases with higher scores reflect higher relevance in the context of the document.
4. Select the top 20 keywords from each answer column to generate questions in such a way that only the multi-word phrases are considered during question creation
5. Create a question template using the most common question verbs and append this question template to the extracted keywords to get valid questions.
6. Create a new column in the existing 

**Data Preprocessing**

The question-answer data is stored in a csv file called 'data_smaller.csv'. To make the text data suitable to be given as input to a deep learning model and also for better information processing, certain preprocessing steps are implemented on the text and are as follows:
1. All the text is converted to lowercase letters to ensure uniformity in text processing
2. Non-alphanumeric characters and punctuation are removed from the text using regex functions
3. The text data is tokenized into words using the word_tokenize library from nltk
4. Stopwords are removed from the text

**Data split**

The dataset is split into train and test sets following an 80:20 ratio for train:test respectively.

**Creating the vocabulary**

Unique tokens are taken from the text and a mapping is created from token: index.
Special tokens are added:
1. vocab[\<pad\>] for padding sequences to the same length
2. vocab[<start>] to initiate the sequence generation
3. vocab['<end>'] to end the sequence generation
4. token-to-ińdex conversion is done to the question tokens and answer tokens columns
5. The max index used in the question and answer is used to set the vocab size

**Model Training**

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

*Code: qa_bot.py*

Training logs:
Epoch [1/100], Loss: 8.846049308776855  
Epoch [2/100], Loss: 8.475971221923828  
Epoch [3/100], Loss: 8.066388130187988  
Epoch [4/100], Loss: 7.60778284072876  
Epoch [5/100], Loss: 7.138458728790283  
Epoch [6/100], Loss: 6.694119453430176  
Epoch [7/100], Loss: 6.279177665710449  
Epoch [8/100], Loss: 5.888763904571533  
Epoch [9/100], Loss: 5.5258917808532715  
Epoch [10/100], Loss: 5.20269250869751  
Epoch [11/100], Loss: 4.935825824737549  
Epoch [12/100], Loss: 4.730057716369629  
Epoch [13/100], Loss: 4.665546417236328  
Epoch [14/100], Loss: 4.58986234664917  
Epoch [15/100], Loss: 4.424426555633545  
Epoch [16/100], Loss: 4.4332451820373535  
Epoch [17/100], Loss: 4.39581823348999  
Epoch [18/100], Loss: 4.306485652923584  
Epoch [19/100], Loss: 4.175334930419922  
Epoch [20/100], Loss: 4.128773212432861  
Epoch [21/100], Loss: 4.102738380432129  
Epoch [22/100], Loss: 4.067449569702148  
Epoch [23/100], Loss: 4.021491050720215  
Epoch [24/100], Loss: 3.973388910293579  
Epoch [25/100], Loss: 3.930445432662964  
Epoch [26/100], Loss: 3.8945937156677246  
Epoch [27/100], Loss: 3.8638901710510254  
Epoch [28/100], Loss: 3.836038827896118  
Epoch [29/100], Loss: 3.8095667362213135  
Epoch [30/100], Loss: 3.783707857131958  
Epoch [31/100], Loss: 3.7582314014434814  
Epoch [32/100], Loss: 3.733102560043335  
Epoch [33/100], Loss: 3.7082653045654297  
Epoch [34/100], Loss: 3.6837310791015625  
Epoch [35/100], Loss: 3.6594812870025635  
Epoch [36/100], Loss: 3.6351733207702637  
Epoch [37/100], Loss: 3.6104519367218018  
Epoch [38/100], Loss: 3.585238456726074  
Epoch [39/100], Loss: 3.559771776199341  
Epoch [40/100], Loss: 3.5340752601623535  
Epoch [41/100], Loss: 3.5077147483825684  
Epoch [42/100], Loss: 3.4805355072021484  
Epoch [43/100], Loss: 3.45280385017395  
Epoch [44/100], Loss: 3.4247844219207764  
Epoch [45/100], Loss: 3.3965542316436768  
Epoch [46/100], Loss: 3.3679730892181396  
Epoch [47/100], Loss: 3.338839054107666  
Epoch [48/100], Loss: 3.3091251850128174  
Epoch [49/100], Loss: 3.2788302898406982  
Epoch [50/100], Loss: 3.24784779548645  
Epoch [51/100], Loss: 3.2160749435424805  
Epoch [52/100], Loss: 3.183650255203247  
Epoch [53/100], Loss: 3.1507387161254883  
Epoch [54/100], Loss: 3.11735463142395  
Epoch [55/100], Loss: 3.083409547805786  
Epoch [56/100], Loss: 3.048855781555176  
Epoch [57/100], Loss: 3.01370906829834  
Epoch [58/100], Loss: 2.977973222732544  
Epoch [59/100], Loss: 2.9416117668151855  
Epoch [60/100], Loss: 2.904789686203003  
Epoch [61/100], Loss: 2.8674418926239014  
Epoch [62/100], Loss: 2.8295743465423584  
Epoch [63/100], Loss: 2.791187047958374  
Epoch [64/100], Loss: 2.752272129058838  
Epoch [65/100], Loss: 2.7128305435180664  
Epoch [66/100], Loss: 2.6728758811950684  
Epoch [67/100], Loss: 2.632448434829712  
Epoch [68/100], Loss: 2.5915586948394775  
Epoch [69/100], Loss: 2.550234794616699  
Epoch [70/100], Loss: 2.508481025695801  
Epoch [71/100], Loss: 2.466334581375122  
Epoch [72/100], Loss: 2.423784017562866  
Epoch [73/100], Loss: 2.3808319568634033  
Epoch [74/100], Loss: 2.3375086784362793  
Epoch [75/100], Loss: 2.293853282928467  
Epoch [76/100], Loss: 2.2498700618743896  
Epoch [77/100], Loss: 2.2055768966674805  
Epoch [78/100], Loss: 2.1610031127929688  
Epoch [79/100], Loss: 2.116163730621338  
Epoch [80/100], Loss: 2.0710737705230713  
Epoch [81/100], Loss: 2.025764226913452  
Epoch [82/100], Loss: 1.9802732467651367  
Epoch [83/100], Loss: 1.9346222877502441  
Epoch [84/100], Loss: 1.8888343572616577  
Epoch [85/100], Loss: 1.8429296016693115  
Epoch [86/100], Loss: 1.7969411611557007  
Epoch [87/100], Loss: 1.75090754032135  
Epoch [88/100], Loss: 1.7048671245574951  
Epoch [89/100], Loss: 1.658852458000183  
Epoch [90/100], Loss: 1.6128982305526733  
Epoch [91/100], Loss: 1.5670435428619385  
Epoch [92/100], Loss: 1.5213292837142944  
Epoch [93/100], Loss: 1.4757944345474243  
Epoch [94/100], Loss: 1.4304802417755127  
Epoch [95/100], Loss: 1.3854341506958008  
Epoch [96/100], Loss: 1.3407020568847656  
Epoch [97/100], Loss: 1.2963279485702515  
Epoch [98/100], Loss: 1.252352237701416  
Epoch [99/100], Loss: 1.2088229656219482  
Epoch [100/100], Loss: 1.1657863855361938  

**QA web service**

Flask is used to deploy the QA bot model as a web service.
The user has to enter the desired question in the box provided to get an answer from the QA bot
The maximum length of the generated answer is set to 50.

*Code: qa_web_service.py*

**Results**

question_tokens:  ['artificial', 'intelligence']
question_indices:  [3813, 3757]

answer_indices:  [5811, 6887, 1838, 3850, 2888, 6762, 631, 5580, 6726, 3850, 2888, 6762, 631, 5580, 6726, 3850, 2888, 6762, 631, 5580, 6726, 3850, 2888, 6762, 631, 5580, 6726, 3850, 2888, 6762, 631, 5580, 6726, 3850, 2888, 6762, 631, 5580, 6726, 3850, 2888, 6762, 631, 5580, 6726, 3850, 2888, 6762, 631, 5580]

The mapping from index to tokens for the answers is still pending as there are some bugs that I need to fix. I am confident that given more time for the project, I would be able to find the root cause of this issue and fix it.






