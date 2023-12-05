import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_accuracy(model, dataloader,device="cpu"):

  with torch.no_grad(): 
    all_predictions = []
    all_labels = []
    correct_predictions = 0
    total_predictions = 0
    model.eval()
    for x,y in dataloader:
      x, y = x.to(device), y.to(device)

      y_hat = model(x)

      predicted = torch.argmax(y_hat,dim=1)

      correct_predictions += (predicted == y).sum().item()
      total_predictions += y.size(0)

      all_predictions.extend(predicted.cpu().numpy())
      all_labels.extend(y.cpu().numpy())
    confusion_mat = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.xlabel('Classe prevista')
    plt.ylabel('Classe verdadeira')
    plt.title('Matriz de Confusão')
    plt.show()

    return correct_predictions/total_predictions

def calc_test_loss(model,dataloader,loss_function,device="cpu"):

  with torch.no_grad():
    model.to(device)
    model.eval()
    total_loss = 0.0
    for x,y in dataloader:
      x, y = x.to(device), y.to(device)
      y_hat = model(x)
      loss = loss_function(y_hat,y)

      total_loss += loss.item()

    return total_loss/len(dataloader)