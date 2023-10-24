import random
import torch

class DLIterator:
    '''Iterates over multiple dataloaders, randomly choosing one at each step.'''
    def __init__(self, dls) -> None:
        self.iters = [iter(dl) for dl in dls]

    def __iter__(self): return self

    def __next__(self):
            for _ in range(len(self.iters)):
                try:
                    it = random.choice(self.iters)
                    return next(it)
                except StopIteration:
                    self.iters.remove(it)
            raise StopIteration


class MutiDataLoader:
    '''Interface combining multiple dataloaders. Used to iterate over dataloaders of different length'''
    def __init__(self,datasets,create_dl_fn) -> None:
        self.dls=list(map(create_dl_fn, datasets))

    def __iter__(self):
        return DLIterator(self.dls)

    def __len__(self):
        return sum(map(len, self.dls))
    
class Batch:
  def __init__(self,x=None,y=None):
    self.x,self.y = x,y

  def to(self,device):
    self.x = self.x.to(device)
    self.y = self.y.to(device)

class HybridBatch(Batch):
  def to(self,device):
    for i in self.x:
      self.x[i] = self.x[i].to(device)
    self.y = self.y.to(device)


def default_collate(x):
  xs = []
  ys = []
  ids = []
  for item in x:
      xs.append(item['x'])
      ys.append(item['y'])
      ids.append(item['id'])
  x = torch.stack(xs)
  y = torch.tensor(ys).unsqueeze(1) # B,1
  batch = Batch(x,y)
  batch.ids = ids
  return batch


def collate_hybrid(x):
  xs_w2v = []
  xs_spctr = []
  ys = []
  ids = []
  for item in x:
      xs_w2v.append(item['x']['w2v2'])
      xs_spctr.append(item['x']['spctr'])
      ys.append(item['y'])
      ids.append(item['id'])
  x_w2v = torch.stack(xs_w2v)
  x_spctr = torch.stack(xs_spctr)
  x = {'w2v2':x_w2v, 'spctr':x_spctr}
  y = torch.tensor(ys).unsqueeze(1) # B,1
  batch = HybridBatch(x,y)
  batch.ids = ids
  return batch


def create_dataloader(ds,bs=64,collate_fn=default_collate):
  return torch.utils.data.DataLoader(ds, bs, shuffle = True, collate_fn=collate_fn)
