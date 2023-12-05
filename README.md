# BertEmbeddings - Neural Network for timelines classification

Neste projeto, foi desenvolvido um modelo de classificação para identificar se as timelines dos usuários são consideradas tóxicas ou não, no contexto das eleições de 2022. A classificação manual demanda tempo e expertise, o que motivou a criação de um sistema automático fundamentado em redes neurais.



## Dados

- Conjunto de dados "less": [https://github.com/SRaimundo/twitter_timeline_classification/blob/main/data/less.csv](link_less)
- Conjunto de dados "more": [https://github.com/SRaimundo/twitter_timeline_classification/blob/main/data/more.csv](link_more)

## Baixando os pesos do modelo treinado

- Modelo treinado para o conjunto de dados "less": [https://drive.google.com/file/d/17M12CFaJEvdS9KeNMFKiM_lXSrb2yXNF/view?usp=sharing](link_model_weights)

---

# Executando o Pré-Processamento
Para executar o script `preProcessing.py`, siga as instruções abaixo.

## Pré-requisitos
Certifique-se de ter Python instalado em seu ambiente. Se ainda não tiver, você pode baixá-lo em [python.org](https://www.python.org/).

## Passos para Execução

1. Navegue até o diretório onde o script está localizado:

    ```bash
    cd caminho/do/repo
    ```

2. Execute o script `preProcessing.py`, fornecendo os seguintes argumentos:

    - `--LESS_FILE`: Caminho do arquivo less.csv.
    - `--MORE_FILE`: Caminho do arquivo more.csv.
    - `--LESS_OUTPUT`: Caminho para salvar os csv processados para less.csv.
    - `--MORE_OUTPUT`: Caminho para salvar os csv processados para more.csv.

    Exemplo:

    ```bash
    python3 preProcessing.py --LESS_FILE caminho/do/seu/less.csv --MORE_FILE caminho/do/seu/more.csv --LESS_OUTPUT caminho/de/saida/less_embeddings.pt --MORE_OUTPUT caminho/de/saida/more_embeddings.pt
    ```

    Substitua os caminhos pelos reais do seu conjunto de dados e destino para os embeddings gerados.

3. Aguarde a conclusão da execução do script.

Pronto! Agora você deve ter seus dados pré-processados e salvos nos caminhos especificados.

---

# Realizando a geração de embeddings

Agora que os dados estão preparados usando o script `preProcessing.py`, siga as instruções abaixo para treinar o modelo com o script `bert_embeddings.py`.

## Passos para Execução

1. Execute o script `bertEmbedddings.py`, fornecendo os seguintes argumentos:

    - `--LESS_FILE`: Caminho do arquivo less.csv.
    - `--MORE_FILE`: Caminho do arquivo more.csv.
    - `--LESS_EMBEDDINGS_FILE`: Caminho para salvar o arquivo less_embeddings.pt.
    - `--MORE_EMBEDDINGS__FILE`: Caminho para salvar o arquivo more_embeddings.pt.

    Exemplo:

    ```bash
    python3 bertEmbedddings.py --LESS_FILE caminho/do/seu/less.csv --MORE_FILE caminho/do/seu/more.csv --LESS_EMBEDDINGS_FILE caminho/do/seu/less.pt --MORE_EMBEDDINGS__FILE caminho/do/seu/more.pt
    ```

    Substitua os caminhos pelos reais do seu conjunto de dados.

3. Aguarde o término a geração de embeddings.

Após a conclusão destes passos, você terá gerados os embeddings para utilização no modelo. Certifique-se de ajustar os caminhos e parâmetros conforme necessário para o seu projeto específico.

# Executando o script train.py

Agora que os embeddings estão preparados usando o script `bert_embeddings.py`, siga as instruções abaixo para treinar o modelo com o script `train.py`.

## Passos para Execução

1. Execute o script `train.py`, fornecendo os seguintes argumentos:

    - `--LESS_EMBEDDINGS`: Caminho do arquivo less_embeddings_file.pt.
    - `--MORE_EMBEDDINGS`: Caminho do arquivo more_embeddings_file.pt.
    - `--WEIGHS_MODEL`: Caminhos para salvar o arquivo com os pesos.

    Exemplo:

    ```bash
    python3 train.py --LESS_FILE caminho/do/seu/less.pt --MORE_FILE caminho/do/seu/more.pt --WEIGHS_MODEL caminho/do/seu/weights.pth 
    ```

    Substitua os caminhos pelos reais do seu conjunto de dados.


3. Aguarde o término do treinamento do modelo.

Após a conclusão destes passos, você terá treinado seu modelo com os embeddings gerados anteriormente. Certifique-se de ajustar os caminhos e parâmetros conforme necessário para o seu projeto específico.

