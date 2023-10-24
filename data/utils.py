import torchaudio
import torch
from functools import partial
from data.dataset import AudioDataset, HybridDataset, FileDataset
from tqdm import tqdm
from typing import Iterable
from math import ceil

def pad_spectrogram(x, maxl):
    lpad = maxl - x.shape[2]
    return torch.cat([x,torch.zeros(1,x.shape[1],lpad)],dim=2)

def create_binned_datasets(data, create_fn) -> list:
  '''Creates datasets of items grouped by length.'''
  dss = []
  for bin in data.bin.unique():
    bin_data = data[data['bin']==bin]
    dss.append(create_fn(bin_data))
  return dss


def get_max_spectrogram_length(data,hop):
  maxl = int(data['samples'].max())
  return maxl//hop + 1


def create_spectrogram_ds(data,norms=None,maxl=None,w=1024,hop=512,n_mels=64,sample_rate=16000):
  if maxl is None: maxl = get_max_spectrogram_length(data,hop)
  pad = partial(pad_spectrogram, maxl=maxl)
  amplitude_to_db = torchaudio.transforms.AmplitudeToDB("magnitude", top_db=80)
  mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=w,
        hop_length=hop,
        n_mels=n_mels)
  transforms = [mel_spectrogram,amplitude_to_db,pad]
  if norms is not None: transforms.append(partial(normalize_trf,mn=norms[0],std=norms[1]))
  return AudioDataset(data,transforms)

def normalize_trf(x,mn,std): return (x-mn)/std

def create_audio_ds(data, norms=None):
  '''Creates a dataset of items in raw audio format'''
  maxl = int(data['samples'].max())
  pad = partial(pad_audio, maxl=maxl)
  transforms = [pad]
  if norms is not None: transforms.append(partial(normalize_trf,mn=norms[0],std=norms[1]))
  ds = AudioDataset(data,transforms)
  return ds

def create_hybrid_datasets_binned(w2v_datasets,spctr_ds):
  '''Creates a hybrid dataset with wav2vec features and spectrograms'''
  dss = []
  for i,ds in enumerate(w2v_datasets):
    spctr_ds_i = spctr_ds[i] if isinstance(spctr_ds,list) else spctr_ds
    dss.append(HybridDataset(ds,spctr_ds_i))
  return dss

def create_file_ds(data,path,transformations=None):
  if transformations is None: transformations = []
  return FileDataset(data,path=path, transformations=transformations)
  

def pad_audio(signal, maxl):
    lpad = maxl - signal.shape[1]
    if lpad > 0:
      signal = torch.nn.functional.pad(signal,(0,lpad),value=0)
    return signal


def compute_norms(datasets:Iterable):
  means = []
  stds = []
  for ds in datasets:
    for id in tqdm(range(len(ds))):
      item = ds[id]
      m = item['x'].mean().item()
      s = item['x'].std().item()
      means.append(m)
      stds.append(s)
  mean = sum(means)/len(means)
  std = sum(stds)/len(stds)
  return mean,std


@torch.no_grad()
def extract_w2v_features(ds,feature_extractor,path,bs,device):
  '''Process raw audio files and extract wav2vec features using CNN feature encoder'''
  feature_extractor = feature_extractor.to(device)
  for i in tqdm(range(0, len(ds), bs)):
    data = ds.annotations[i:i+bs]
    items = [ds[k]['x'] for k in range(i,i+bs,1) if k < len(ds)]
    tensor = torch.cat(items).to(device)
    features = feature_extractor(tensor).transpose(1, 2)
    for i,fn in enumerate(data.index):
      torch.save(features[i].cpu(), f'{path}/{fn}.pt')