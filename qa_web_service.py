from flask import Flask, render_template, request
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from model import Sequence2Sequence
from qa_bot import vocab_size, embedding_dim, hidden_dim, preprocess_text, model_save_path, vocab



app = Flask(__name__)
model_save_path = '/home/yadu/MercedesBenz/trained_qa_model_100epochs.pth'
# Load the trained model and vocabulary
model = Sequence2Sequence(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load(model_save_path))
model.eval()


app = Flask(__name__)


# Vocabulary for text preprocessing
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        question = request.form['question']
        tokenized_question = preprocess_text(question)
        question_indices = [vocab[token] for token in tokenized_question]
        print(tokenized_question)
        current_token = torch.LongTensor([vocab['<start>']]).unsqueeze(0)
        question_tensor = torch.LongTensor(question_indices).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            answer_indices = []

            # Generate answer using greedy decoding
            max_len = 50  # Maximum length for generated answer

            for _ in range(max_len):
                output = model(question_tensor, current_token)
                #output = loaded_model(question_tensor)
                _, predicted_token = torch.max(output[:, -1, :], dim=1)
                answer_indices.append(predicted_token.item())
                #print("answer_indices: ", answer_indices)

                if predicted_token.item() == vocab['<end>']:
                    break

                current_token = predicted_token.unsqueeze(0)


            # Convert answer indices back to tokens using vocabulary mapping
            answer_tokens = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in answer_indices]
            answer = ' '.join(answer_tokens).replace('<start>', '').replace('<end>', '').strip()
            print("answer: ", answer)
        return render_template('index.html', question=question, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)

