import copy
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import json

class CustomDataset(Dataset):
    def __init__(self, path):

        self.max_m = 100
        filelist = glob.glob(path + '/*.json')
        print(len(filelist))
        self.data_list = []
        self.n_list = []
        self.mms_list = []
        self.max_val_list = []
        for filename in filelist:
            n,m,max_val = get_params_from_filename(filename)
            with open(filename) as jsonFile:
                data_list_in_file = json.load(jsonFile)
                for example in data_list_in_file:
                    values = example[0]["values"]
                    values_copy = copy.deepcopy(values)
                    self.zero_pad(values_copy)

                    # fixing 'leftover' decimal errors from the linear program solvers
                    #if not example[0]["mms"].is_integer():

                    example[0]["mms"] = round(example[0]["mms"])

                    self.mms_list.append(example[0]["mms"])
                    values_copy.sort(reverse=True)
                    self.data_list.append(copy.deepcopy(values_copy))
                    self.n_list.append(float(n))
                    self.max_val_list.append(max_val)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        normalized_values, normalized_mms = self.normalize_entry(idx)
        x = torch.FloatTensor([self.n_list[idx]]+normalized_values)
        return x, normalized_mms

    def normalize_entry(self, idx):
        # normalizing entry, by deviding each value by the total sum of the values
        val_sum = sum(self.data_list[idx])
        if val_sum == 0:
            normalized_values = [float(elem) for elem in self.data_list[idx]]
            normalized_mms = self.mms_list[idx]
        else:
            normalized_values = [float(elem / val_sum) for elem in self.data_list[idx]]
            normalized_mms = self.mms_list[idx] / val_sum
        return normalized_values, normalized_mms

    def zero_pad(self, values):
        m = len(values)
        values += [0]*(self.max_m-m)

def get_params_from_filename(filename):

        n_str, m_str, max_v_str = filename[len('Dataset\\'):].strip('_uniform.json').split('_')
        n, m, max_v = int(n_str), int(m_str), int(max_v_str)
        return n, m, max_v