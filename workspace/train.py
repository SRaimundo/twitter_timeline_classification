from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from dataset import ClassificationDataset
from model import BertFC
from evaluate import evaluate_model_accuracy, calc_test_loss
import argparse
import sys

def optimize(model,train_loader,test_loader,learnig_rate=0.001,num_epochs=10,device="cpu",outpath='tweetClassification.pth'):
    train_losses = []
    test_losses = []

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learnig_rate)

    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0

        for embeddings,labels in train_loader :
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(embeddings)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss/len(train_loader))
        average_loss_test = calc_test_loss(model=model,dataloader=test_loader,loss_function=loss_function,device=device)
        test_losses.append(average_loss_test)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}         Test Loss: {average_loss_test}")

    torch.save(model.state_dict(), outpath)
    return train_losses, test_losses



def main():

    parser = argparse.ArgumentParser(description="Treinamento do modelo")
    parser.add_argument("--LESS_FILE", type=str, help="Caminho para o arquivo less.csv")
    parser.add_argument("--MORE_FILE", type=str, help="Caminho para o arquivo more.csv")
    parser.add_argument("--WEIGHS_MODEL", type=str, help="Caminho para salvar os pesos do modelo")

    args = parser.parse_args()

    if not all([args.LESS_FILE, args.MORE_FILE, args.WEIGHS_MODEL]):
        print("Usage: python3 arquivo.py --LESS_FILE <caminho_less.csv> --MORE_FILE <caminho_more.csv> --WEIGHS_MODEL <caminho_pesos_modelo.pth>")
        sys.exit(1)

    less_embeddings_file = args.LESS_FILE
    more_embeddings_file = args.MORE_FILE
    save_weights = args.WEIGHS_MODEL
  
 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    less_embeddings = torch.load(less_embeddings_file ,map_location=torch.device('cpu'))
    less_embeddings = torch.mean(less_embeddings, dim=1)

    more_embeddings = torch.load(more_embeddings_file,map_location=torch.device('cpu'))
    more_embeddings = torch.mean(more_embeddings, dim=1)

    torch.manual_seed(4)

    indices = torch.randperm(len(less_embeddings))[:more_embeddings.size(0)]

    # Selecione os elementos correspondentes aos índices gerados
    selected_embeddings = [less_embeddings[i] for i in indices]

    less = torch.stack(selected_embeddings)
    # less = less_embeddings
    # more = torch.stack(more_embeddings)
    more = more_embeddings

    labels1 = torch.zeros(less.size(0),dtype=torch.long)
    labels2 = torch.ones(more.size(0),dtype=torch.long)

    embeddings = torch.cat((less,more),dim=0)
    labels = torch.cat((labels1,labels2),dim=0)

    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )

    train_dataset = ClassificationDataset(train_embeddings,train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = ClassificationDataset(test_embeddings,test_labels)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

    torch.manual_seed(4)
    embedding_dim = embeddings.size(1)
    output_size = 2
    dropout_rate = 0.3
    num_epochs = 14
    

    modelBF = BertFC(embedding_dim,output_size,dropout_rate)
    train_losses, test_losses = optimize(modelBF,train_loader,test_loader,num_epochs=num_epochs,device=device,outpath=save_weights)


    # Plot the learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b',label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, marker='x', linestyle='-', color='g',label='Test Loss')

    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.show()
    
    modelBF.eval()

    train_accuracy  = evaluate_model_accuracy(modelBF, train_loader,device=device)
    test_accuracy = evaluate_model_accuracy(modelBF, test_loader,device=device)

    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
   main()