import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
import numpy as np
import json

class IEMOCAPDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
         self.trainVid, self.testVid = pickle.load(open('./IEMOCAP.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        del self.videoSpeakers['Ses05F_script02_2']
        del self.videoLabels['Ses05F_script02_2']
        self.testVid.remove('Ses05F_script02_2')
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]





class MELDDataset(Dataset):

    def __init__(self, path, n_classes, train=True):
        self.videoIDs, self.videoSpeakers, self.videoText, \
        self.trainVid, self.testVid, self.videoLabels = pickle.load(open('./MELD.pkl', 'rb'))
        del self.videoSpeakers[1432]
        del self.videoLabels[1432]
        self.testVid.remove(1432)
        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]

        return torch.FloatTensor(self.videoText[vid]),torch.FloatTensor(self.videoSpeakers[vid]),torch.FloatTensor([1] * len(self.videoLabels[vid])),torch.LongTensor(self.videoLabels[vid]),vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]