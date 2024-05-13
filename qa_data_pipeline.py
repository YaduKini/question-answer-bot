import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd


# List of URLs to scrape
urls = ['https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning',
        'https://en.wikipedia.org/wiki/Machine_learning',
        'https://en.wikipedia.org/wiki/Soft_computing',
        'https://en.wikipedia.org/wiki/Weak_artificial_intelligence',
        'https://en.wikipedia.org/wiki/Turing_test']
        # 'https://en.wikipedia.org/wiki/Information_theory',
        # 'https://en.wikipedia.org/wiki/History_of_artificial_intelligence',
        # 'https://en.wikipedia.org/wiki/Regulation_of_algorithms',
        # 'https://en.wikipedia.org/wiki/Regulation_of_artificial_intelligence',
        # 'https://en.wikipedia.org/wiki/Mistral_AI',
        # 'https://en.wikipedia.org/wiki/LLaMA',
        # 'https://en.wikipedia.org/wiki/Meta_Platforms',
        # 'https://en.wikipedia.org/wiki/EleutherAI',
        # 'https://en.wikipedia.org/wiki/Hugging_Face',
        # 'https://en.wikipedia.org/wiki/Friendly_artificial_intelligence',
        # 'https://en.wikipedia.org/wiki/Moral_agency#Artificial_moral_agents',
        # 'https://en.wikipedia.org/wiki/AI_safety',
        # 'https://en.wikipedia.org/wiki/Machine_ethics',
        # 'https://en.wikipedia.org/wiki/Existential_risk_from_artificial_general_intelligence',
        # 'https://en.wikipedia.org/wiki/Workplace_impact_of_artificial_intelligence',
        # 'https://en.wikipedia.org/wiki/Multi-task_learning',
        # 'https://en.wikipedia.org/wiki/DeepDream',
        # 'https://en.wikipedia.org/wiki/Algorithmic_transparency',
        # 'https://en.wikipedia.org/wiki/Explainable_artificial_intelligence',
        # 'https://en.wikipedia.org/wiki/Fairness_(machine_learning)',
        # 'https://en.wikipedia.org/wiki/Recommender_system',
        # 'https://en.wikipedia.org/wiki/Ethics_of_artificial_intelligence',
        # 'https://en.wikipedia.org/wiki/Stable_Diffusion',
        # 'https://en.wikipedia.org/wiki/Text-to-image_model',
        # 'https://en.wikipedia.org/wiki/Generative_artificial_intelligence',
        # 'https://en.wikipedia.org/wiki/Vehicular_automation',
        # 'https://en.wikipedia.org/wiki/MuZero',
        # 'https://en.wikipedia.org/wiki/Google_DeepMind',
        # 'https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer)',
        # 'https://en.wikipedia.org/wiki/Artificial_intelligence_in_video_games',
        # 'https://en.wikipedia.org/wiki/Artificial_intelligence_in_healthcare',
        # 'https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback',
        # 'https://en.wikipedia.org/wiki/Large_language_model',
        # 'https://en.wikipedia.org/wiki/Generative_pre-trained_transformer',
        # 'https://en.wikipedia.org/wiki/Digital_image_processing',
        # 'https://en.wikipedia.org/wiki/Convolutional_neural_network',
        # 'https://en.wikipedia.org/wiki/Perceptron',
        # 'https://en.wikipedia.org/wiki/Long_short-term_memory',
        # 'https://en.wikipedia.org/wiki/Recurrent_neural_network',
        # 'https://en.wikipedia.org/wiki/Backpropagation',
        # 'https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm',
        # 'https://en.wikipedia.org/wiki/Bayesian_inference',
        # 'https://en.wikipedia.org/wiki/Bayesian_network',
        # 'https://en.wikipedia.org/wiki/Game_theory',
        # 'https://en.wikipedia.org/wiki/Influence_diagram',
        # 'https://en.wikipedia.org/wiki/Markov_decision_process',
        # 'https://en.wikipedia.org/wiki/Value_of_information',
        # 'https://en.wikipedia.org/wiki/Decision_analysis',
        # 'https://en.wikipedia.org/wiki/Decision_theory',
        # 'https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms',
        # 'https://en.wikipedia.org/wiki/Particle_swarm_optimization',
        # 'https://en.wikipedia.org/wiki/Swarm_intelligence',
        # 'https://en.wikipedia.org/wiki/Evolutionary_computation',
        # 'https://en.wikipedia.org/wiki/Gradient_descent',
        # 'https://en.wikipedia.org/wiki/Mathematical_optimization',
        # 'https://en.wikipedia.org/wiki/Computer_vision',
        # 'https://en.wikipedia.org/wiki/Machine_perception',
        # 'https://en.wikipedia.org/wiki/Question_answering',
        # 'https://en.wikipedia.org/wiki/Information_retrieval',
        # 'https://en.wikipedia.org/wiki/Information_extraction',
        # 'https://en.wikipedia.org/wiki/Machine_translation',
        # 'https://en.wikipedia.org/wiki/Speech_synthesis',
        # 'https://en.wikipedia.org/wiki/Speech_recognition',
        # 'https://en.wikipedia.org/wiki/Natural_language_processing',
        # 'https://en.wikipedia.org/wiki/Transfer_learning',
        # 'https://en.wikipedia.org/wiki/Reinforcement_learning',
        # 'https://en.wikipedia.org/wiki/Regression_analysis',
        # 'https://en.wikipedia.org/wiki/Statistical_classification',
        # 'https://en.wikipedia.org/wiki/Supervised_learning',
        # 'https://en.wikipedia.org/wiki/Unsupervised_learning']


# function to get data from url
def get_data_from_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text_content = ' '.join(para.get_text() for para in paragraphs)
    return text_content


# get text content from each url and write to csv file
with open('data_smaller.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['answer'])
    for url in urls:
        text_content = get_data_from_webpage(url)
        writer.writerow([text_content])



df = pd.read_csv('data_smaller.csv')

print(df.columns)

