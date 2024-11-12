from tqdm import tqdm
import torchaudio
import torch


class BasicDataset:
  def __init__(self,annotations,transformations):
    self.annotations = annotations
    self.transformations = transformations
    self.cache = dict()

  def __getitem__(self, index):
    id = self.get_id(index)
    x = self.get_x(id)
    y = self.get_y(id)
    return {'x':x,'y':y,'id':id}

  def get_id(self,index):
    '''Get a unique item identifier'''
    return self.annotations.iloc[index].name

  def __len__(self):
    return len(self.annotations)

  def get_y(self,id):
    return self.annotations.loc[id]['is_active']

  def get_x(self,id):
    x = self.cache.get(id, None)
    if x is None:
      x = self.load(id)
      x = self.transform(x)
    return x

  def transform(self, signal):
      for tr in self.transformations:
        signal = tr(signal)
      return signal

  def load(self, id): pass

  def prefetch(self, frac):
    '''Prefetches the fraction of the dataset'''
    n = round(len(self)*frac)
    for id in tqdm(self.annotations.index[:n],position=0, leave=True):
      self.cache[id] = self.get_x(id)



class AudioDataset(BasicDataset):
  '''Dataset of items based on the raw audio files'''
  def __init__(self,annotations,transformations,path='D:/FYP/Audios/Audio'):
    super().__init__(annotations,transformations)
    self.path = path

  def load(self,id):
    path = f'{self.path}/{id}'
    signal,_ = torchaudio.load(path)
    return signal
  

class FileDataset(BasicDataset):
  '''Dataset of items based on the .pt files saved on disk'''
  def __init__(self,annotations,transformations,path):
    super().__init__(annotations, transformations=transformations)
    self.path = path

  def load(self,id):
    path = f'{self.path}/{id}.pt'
    return torch.load(path)
  

class HybridDataset(BasicDataset):
    '''Contains spectrograms and wav2vec features for each item'''
    def __init__(self, w2v_ds, spctr_ds):
      super().__init__(w2v_ds.annotations, transformations=[])
      self.w2v_ds = w2v_ds
      self.spctr_ds = spctr_ds

    def get_x(self,id):
      return {'w2v2':self.w2v_ds.get_x(id), 'spctr':self.spctr_ds.get_x(id)}

    def prefetch(self, frac=1):
      self.spctr_ds.prefetch(frac) # prefetch only spectrograms, w2v features are too heavy