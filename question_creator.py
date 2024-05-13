import pandas as pd
import spacy
from rake_nltk import Rake
import random
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')



def remove_stopwords(sentences):
    stop_words = set(stopwords.words('english'))

    filtered_sentences = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)

        # Filter out stopwords from the sentence
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Join the filtered words back into a sentence
        filtered_sentence = ' '.join(filtered_words)
        
        # Append the filtered sentence to the result list
        filtered_sentences.append(filtered_sentence)

    return filtered_sentences


# Function to generate questions using keyword extraction and templates
def generate_questions(text):
    # Extract keywords using RAKE (Rapid Automatic Keyword Extraction)
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()

    #print("len of keywords: ", len(keywords))
    # Generate questions from top-ranked keywords
    questions = []
    for keyword in keywords[:20]:  
        doc = nlp(keyword)
        if doc and len(doc) > 1:  # Process only multi-word phrases
            subject = doc[0].text
            verb = random.choice(["is", "was", "does"])  # Randomly select a verb for question template
            template = f"What {verb} {keyword}?"
            questions.append((template, text))
    
    return questions


def preprocess_text(text):
    # remove non alphanumeric characters

    print("type of text: ", type(text))
    print("len of text: ", len(text))

    # remove non alphanumeric characters
    sent = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # remove punctuations
    sent = re.sub(r'[^\w\s]', '', sent)

    # remove all empty spaces in text 
    #sent = re.sub(r'\s+', '', sent)

    # convert all text to lower case
    sent = sent.lower()

    print(sent)


    # tokenize text into sentences
    tokenized_sentences = nltk.sent_tokenize(sent)

    #print("tokenized sentences: ", tokenized_sentences)

    filtered_tokenized_sents = remove_stopwords(tokenized_sentences)

    print("filtered_tokenized_sents sentences: ", filtered_tokenized_sents)



    # Remove stopwords from each tokenized sentence
    # filtered_sentences = []
    # for sentence_tokens in tokenized_sentences:
    #     filtered_tokens = [token for token in sentence_tokens if token.lower() not in stopwords_set]
    #     filtered_sentences.append(filtered_tokens)


    return tokenized_sentences


#df = pd.read_csv('data.csv')
df = pd.read_csv('data_smaller.csv')

print(df.shape)

# Extract keywords and generate questions using optimized pandas operations

df['question'] = df['answer'].apply(lambda x: generate_questions(x))

print(df.columns)

print(df.head())
print(df.columns)
    
print(df.shape)
df.to_csv('question_answer_smaller.csv', index=True)

