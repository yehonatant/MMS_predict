from torch.utils.data import Dataset, DataLoader
from SimpleDataLoader import CustomDataset
path_to_training_data = 'Dataset'

training_ds = CustomDataset(path_to_training_data)
print(len(training_ds))
# some_random_idx = 52
# print(training_ds[some_random_idx])

# z = 0.003
# t = float(round(z))
# print(t,type(t))



training_dataloader = DataLoader(training_ds,batch_size=300,shuffle=True)

for x,y in training_dataloader:
    print(x.shape,y.shape)
    break