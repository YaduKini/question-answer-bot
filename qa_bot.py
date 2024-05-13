import pandas as pd
import torch
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import requests
from bs4 import BeautifulSoup
from model import Sequence2Sequence
import torch.nn as nn


# physicsqa csv path
#csv_path = '/home/yadu/MercedesBenz/physicsqa.csv'
csv_path = 'question_answer_smaller.csv'


# load physicsqa csv
df = pd.read_csv(csv_path)

#tokenizer = nltk.tokenize.WordPunctTokenizer()


def tokens_to_indices(tokens):
    return [vocab[token] for token in tokens]


def get_wiki_data(url):
    """
    
    
    """

    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all <p> tags
        paragraphs = soup.find_all('p')
        prettyHTML = soup.prettify()
        #print(type(paragraphs))

        paragraphs_list = []

        for p in paragraphs:
            para = p.get_text()
            #print(para)
            paragraphs_list.append(para)

        
        corpus = '\n'.join(paragraphs_list)

        clean_corpus = preprocess_text(corpus)

    
    return clean_corpus



# def tokenize_and_remove_stopwords(text):
#     """
#     """

#     tokens = tokenizer.tokenize(text)

#     stop_words = set(stopwords.words('english'))

#     # Remove stopwords
#     tokens = [token for token in tokens if token.lower() not in stop_words]

#     return tokens



#print(df.head())

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Join tokens back into a string
    #preprocessed_text = ' '.join(tokens)

    return tokens


# Apply preprocessing to question column
df['question_tokens'] = df['question'].apply(preprocess_text)

# Apply preprocessing to answer column
df['answer_tokens'] = df['answer'].apply(preprocess_text)

# Display the preprocessed DataFrame
print(df.head())
print(df.columns)

# Split into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("len of train df: ", len(train_df))
print("len of test_df: ", len(test_df))

#tokenizer = get_tokenizer('basic_english')

# Tokenize text data
#train_df['question_tokens'] = train_df['question_preprocessed'].apply(preprocess_text)
#train_df['answer_tokens'] = train_df['answer_preprocessed'].apply(preprocess_text)

#print("train df head: ", train_df.head())
print("train df cols: ", train_df.columns)

# train tokens
all_train_tokens = train_df['question_tokens'].explode().unique().tolist() + train_df['answer_tokens'].explode().unique().tolist()

print("len of train tokens: ", len(all_train_tokens))

# Create vocabulary mapping (token to index)
vocab = {token: idx + 1 for idx, token in enumerate(all_train_tokens)}  # Start index from 1 (reserve 0 for padding)

# Add special tokens to the vocabulary if needed
vocab['<pad>'] = 0  # Padding token (for sequences of variable length)
vocab['<start>'] = len(vocab)  # Start token (for sequence generation)
vocab['<end>'] = len(vocab) + 1  # End token (for sequence generation)

print(vocab)
# wikipedia page used for getting text corpus
#url = "https://en.wikipedia.org/wiki/Physics"

#clean_corpus = get_wiki_data(url)


# # Convert tokens to sequences of indices
# train_df['question_indices'] = train_df['question_tokens'].apply(lambda x: [tokenizer.vocab[token] for token in x])
# train_df['answer_indices'] = train_df['answer_tokens'].apply(lambda x: [tokenizer.vocab[token] for token in x])

# Apply token-to-index conversion to question_tokens and answer_tokens columns
train_df['question_indices'] = train_df['question_tokens'].apply(tokens_to_indices)
train_df['answer_indices'] = train_df['answer_tokens'].apply(tokens_to_indices)

print("train df head: ", train_df.head())

# Convert to PyTorch tensors
questions = [torch.LongTensor(q) for q in train_df['question_indices']]
answers = [torch.LongTensor(a) for a in train_df['answer_indices']]

# Pad sequences to the same length
questions_padded = pad_sequence(questions, batch_first=True)
answers_padded = pad_sequence(answers, batch_first=True)

max_index_input = torch.max(questions_padded).item()
max_index_target = torch.max(answers_padded).item()

print("max_index_input: ", max_index_input)
print("max_index_target: ", max_index_target)

# Determine the overall maximum index used
max_index_used = max(max_index_input, max_index_target)

print("max_index_used: ", max_index_used)

vocab_size = max_index_used + 1

print("questions_padded shape: ", questions_padded.shape)
print("answers_padded shape: ", answers_padded.shape)

#print("answers_padded[:,:,-1]: ", answers_padded[:,:,-1])

#vocab_size = len(vocab)
#vocab_size = len(vocab) + 1
print("vocab size: ", vocab_size)

# Define hyperparameters
embedding_dim = 128
hidden_dim = 256
learning_rate = 0.001
num_epochs = 1

# Instantiate the model
model = Sequence2Sequence(vocab_size, embedding_dim, hidden_dim)

print("running till model")

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    #output = model(questions_padded, answers_padded[:,:,-1])
    output = model(questions_padded, answers_padded)
    output_dim = output.size(2)
    output = output.view(-1, output_dim)
    #output = output.reshape(-1, output.shape[2])
    target = answers_padded.view(-1)
    #target = answers_padded[:, 1:].reshape(-1)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# Define path to save the trained model
model_save_path = '/home/yadu/MercedesBenz/trained_qa_model_100epochs.pth'

# Save the model's state dictionary to the specified path
torch.save(model.state_dict(), model_save_path)



def predict_answer(loaded_model, question, vocab):
    loaded_model.eval()
    #print("len of vocab: ", len(vocab))

    with torch.no_grad():
        # Preprocess the input question
        question_tokens = preprocess_text(question)
        print("question_tokens: ", question_tokens)
        question_indices = [vocab[token] for token in question_tokens]
        print("question_indices: ", question_indices)
        question_tensor = torch.LongTensor(question_indices).unsqueeze(0)
        print("question_tensor: ", question_tensor)

        # Initialize the current token for answer generation
        current_token = torch.LongTensor([vocab['<start>']]).unsqueeze(0)
        print("current token: ", current_token)
        answer_indices = []

        # Generate answer using greedy decoding
        max_len = 50  # Maximum length for generated answer

        for _ in range(max_len):
            output = loaded_model(question_tensor, current_token)
            _, predicted_token = torch.max(output[:, -1, :], dim=1)
            answer_indices.append(predicted_token.item())
            print("answer_indices: ", answer_indices)

            if predicted_token.item() == vocab['<end>']:
                break

            current_token = predicted_token.unsqueeze(0)

        # Convert answer indices back to tokens using vocabulary mapping
        # answer_tokens = [token for idx, token in vocab.items() if idx in answer_indices]
        # print("answer tokens: ", answer_tokens)
        # answer = ' '.join(answer_tokens).replace('<start>', '').replace('<end>', '').strip()
        # print("answer: ", answer)

        # Convert answer indices back to tokens using vocabulary mapping
        answer_tokens = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in answer_indices]
        answer = ' '.join(answer_tokens).replace('<start>', '').replace('<end>', '').strip()

        print("Answer Tokens:", answer_tokens)
        print("Predicted Answer:", answer)


    return answer

# Example usage
#new_question = 'what is artificial intelligence?'

# Load the trained model and vocabulary
#loaded_model = Sequence2Sequence(vocab_size, embedding_dim, hidden_dim)
#loaded_model.load_state_dict(torch.load(model_save_path))

# Perform prediction
#predicted_answer = predict_answer(loaded_model, new_question, vocab)

#print(f'Predicted Answer: {predicted_answer}')







