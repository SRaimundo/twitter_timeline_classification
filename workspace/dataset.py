from torch.utils.data import Dataset, DataLoader

class ClassificationDataset(Dataset):
  def __init__(self,embeddings,labels):
    self.embeddings = embeddings
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    embedding = self.embeddings[idx]
    label = self.labels[idx]

    return embedding, label