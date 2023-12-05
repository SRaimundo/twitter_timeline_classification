import pandas as pd
import torch

# Text cleaning
import re
import emoji
import nltk
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Baixar o recurso necessário para a tokenização (pode precisar ser executado apenas uma vez)
nltk.download('punkt')

# Baixar o recurso necessário para o pré-processamento
nltk.download('stopwords')

nlp = spacy.load("pt_core_news_sm")

device = "cuda" if torch.cuda.is_available() else "cpu"

#Remover tweets vazios
def clean_empty(df):
  empty = df['tweet'].apply(len).apply(lambda x: x > 0)
  print(f"Quantidade de tweets vazios encontrados: {empty.value_counts().get(False, 0)}")

  # Removendo linhas com quantidade de caracteres igual a 0
  return df[empty]


def data_cleaning(tweet):
    # Remoção de emoji
    tweet = emoji.demojize(tweet, delimiters=(' :', ': '), language='pt')
    tweet = re.sub(r'(?<=\s:)(.*?)(?=:|\s|$)', '', tweet)
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r';', '', tweet)

    tweet = re.sub(r'\r|\n', ' ', tweet.lower())  # Replace newline and carriage return with space, and convert to lowercase
    tweet = re.sub(r"(?:\@|https?\://)\S+", "", tweet)  # Remove links and mentions
    tweet = re.sub(r'rt', '', tweet)
    tweet = re.sub(r"\s\s+", " ", tweet)

    words = tweet.split()
    return tweet if len(words) > 3 else ""


#@title Função de lematização usando spacy
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

#@title Remoção de stop-words
def remove_stopwords(text):
  tokens = word_tokenize(text.lower())
  stop_words = set(stopwords.words('portuguese'))  # Escolha o idioma apropriado
  tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
  return ' '.join(tokens)



less_df = pd.read_csv('less.csv', delimiter=';', skiprows=0, low_memory=False)
more_df = pd.read_csv('more.csv', delimiter=';', skiprows=0, low_memory=False)

#Aplicando limpeza
less_df['tweet'] = [data_cleaning(tweet) for tweet in less_df['tweet']]
less_df = clean_empty(less_df)

more_df['tweet'] = [data_cleaning(tweet) for tweet in more_df['tweet']]
more_df = clean_empty(more_df)

#@title Aplicando remoção de stop-words nos dados
more_df['tweet'].apply(remove_stopwords)
less_df['tweet'].apply(remove_stopwords)

#@title Removendo tweets vazios após a remoção de stop-words
more_df = clean_empty(more_df)
less_df = clean_empty(less_df)

less_df.to_csv('less_df.csv', sep=';')
more_df.to_csv('more_df.csv', sep=';')