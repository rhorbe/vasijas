
from torchvision import datasets
from PIL import Image
from scipy import stats
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CustomData(Dataset):
    """
    CustomData dataset
    """

    def __init__(self, dirpath_csv, type_frag, complete=True, part_frag='', transform=None):
        super().__init__()
        self.dirpath_csv = dirpath_csv
        self.transform_input = transform
        self.complete = complete
        self.csv = pd.read_csv(dirpath_csv, index_col=0)
        self.type_frag = type_frag
        self.part_frag = part_frag
    
    def getImage(self, ids_name):
                  
        if(self.complete):
            root_complete = 'data/perfiles_CATA/png/{}.png'.format(ids_name)  
            return Image.open(root_complete)
        else:
            root_incomplete = 'data/perfiles_CATA/png/{}_{}.png'.format(ids_name, self.part_frag)
            return Image.open(root_incomplete)


    def __len__(self):
        return self.csv.shape[0]
    
    def ids_names(self):
        return np.array(self.csv[['Identificador']]).flatten()

    def __getitem__(self, index):
        row = self.csv.iloc[int(index)]
        ids_name = row['Identificador']
        class_filename = row[self.type_frag]
        try:
            
            img = self.getImage(ids_name)
            
        except FileNotFoundError:
            img =  Image.new('RGB', (128, 128), color = 'white')
            
        if self.transform_input is not None:
            img = self.transform_input(img)            
        
        return img, int(class_filename)-1
