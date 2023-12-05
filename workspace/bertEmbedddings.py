from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
import torch
import pandas as pd
import sys
import argparse 

class BertEmbeddings():
    def __init__(self,bertModel,tokenizerModel,device):
        self.device = device
        self.model = AutoModelForPreTraining.from_pretrained(bertModel)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizerModel, do_lower_case=False)
    
    def apllyPadding(self,embeddings,max_embeddings):
        new_embeddings = []
        for embedding in embeddings:
            padding_size = max_embeddings - embedding.size(0)
            if padding_size == 0:
                new_embeddings.append(embedding)
                continue
            dim = embedding.size(1)
            zeros = [torch.zeros(dim) for i in range(padding_size)]
            zeros = torch.stack(zeros)
            new_embeddings.append(torch.cat([embedding,zeros],dim=0))
        return new_embeddings

    
    def generateEmbeddings(self,sample,padding=False): #semple is a dictinary(key: author_id, value: list of tweets)
        embeddings = []
        max_embeddings = 0

        for author_id, tweet in sample.items():
            segment_embeddings = []

            for input in tweet:
                input_ids = self.tokenizer.encode(input, return_tensors='pt').to(self.device)

                with torch.no_grad():
                    outs = self.model(input_ids)
                    encoded = outs[0][0,1:-1]

                segment_embeddings.append(encoded.mean(dim=0))

            if(segment_embeddings):
                segment_embeddings = torch.stack(segment_embeddings)
                embeddings.append(segment_embeddings)
                max_embeddings = max(max_embeddings, segment_embeddings.size(0))
            
        if padding:
            embeddings = self.apllyPadding(embeddings,max_embeddings)


        return embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Geração de embeddings usando BERT")
    parser.add_argument("--LESS_FILE", type=str, help="Caminho para o arquivo less.csv")
    parser.add_argument("--MORE_FILE", type=str, help="Caminho para o arquivo more.csv")
    parser.add_argument("--LESS_EMBEDDINGS_FILE", type=str, help="Caminho para salvar o arquivo less_embeddings.pt")
    parser.add_argument("--MORE_EMBEDDINGS__FILE", type=str, help="Caminho para salvar o arquivo more_embeddings.pt")

    args = parser.parse_args()

    if not all([args.LESS_FILE, args.MORE_FILE, args.LESS_EMBEDDINGS_FILE, args.MORE_EMBEDDINGS__FILE]):
        print("Usage: python3 bertEmbeddings.py --LESS_FILE <caminho_less.csv> --MORE_FILE <caminho_more.csv> ")
        sys.exit(1)

    less_df_file = args.LESS_FILE
    more_df_file = args.MORE_FILE
    less_pt = args.LESS_EMBEDDINGS_FILE
    more_pt = args.MORE_EMBEDDINGS__FILE

    #@title Função para criar dict
    def criar_dict(df):
        # Ordenando o DataFrame por 'author_id' e 'data'
        df_sorted = df.sort_values(by=['author_id', 'created_at'])

        tweets_dict = {}
        # Itera sobre as linhas do DataFrame
        for index, row in df.iterrows():
            author_id = row['author_id']
            tweet = row['tweet']

            # Verifica se o author_id já está no dicionário
            if author_id in tweets_dict:
                # Adiciona o tweet ao vetor de tweets do autor
                tweets_dict[author_id].append(tweet)
            else:
                # Cria um novo vetor de tweets para o autor
                tweets_dict[author_id] = [tweet]

        return tweets_dict
    
    #@title Função para remover duplicados e selecionar amostra
    def remover_duplicados(tweets_por_autor, max_tweets = 32):
        # Itera sobre os tweets agrupados por autor
        for author_id, tweets in tweets_por_autor.items():
            # Remove tweets duplicados
            tweets_por_autor[author_id] = list(set(tweets))

            # Limita o vetor a 32 tweets pegando os últimos 32
            tweets_por_autor[author_id] = tweets_por_autor[author_id][-max_tweets:]


        return tweets_por_autor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertModel = 'neuralmind/bert-base-portuguese-cased'
    tokenizerModel = 'neuralmind/bert-base-portuguese-cased'

    model = BertEmbeddings(bertModel,tokenizerModel,device)

    less_df = pd.read_csv(less_df_file, delimiter=';', low_memory=False)
    more_df = pd.read_csv(more_df_file, delimiter=';', skiprows=0, low_memory=False)

    #@title Gerar dicionário
    more_dict = remover_duplicados(criar_dict(more_df))
    less_dict = remover_duplicados(criar_dict(less_df))

    less_embeddings = model.generateEmbeddings(less_dict,True)
    more_embeddings = model.generateEmbeddings(more_dict,True)

    #@title Salvar objetos
    torch.save(torch.stack(less_embeddings),less_pt)
    torch.save(torch.stack(more_embeddings),more_pt)



