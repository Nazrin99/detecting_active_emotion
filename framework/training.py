from tqdm import tqdm
import functools
from typing import Iterable
from tqdm import tqdm
import torch


class Listener:
  '''A callback that can be registered to a trainer to be called at different stages of the training'''
  _order = 0
  def before_fit(self): pass
  def before_batch(self): pass
  def after_batch(self): pass
  def after_epoch(self): pass
  def before_epoch(self): pass
  def after_fit(self): pass
  def before_backward(self): pass
  def before_optim(self): pass
  def register_trainer(self,trainer): self.trainer = trainer


class ListenerList(Listener):
    '''A container of callbacks to delegete calls to all the listeners'''
    def __init__(self, callbacks: Iterable[Listener], trainer = None):
      self.callbacks = sorted(callbacks, key=lambda x: x._order)
      if trainer is not None:
        self.register(trainer)

    def register(self, trainer):
      for obs in self.callbacks: obs.register_trainer(trainer)

    def __getattribute__(self, attr):
      if hasattr(Listener, attr): # redirect call to all childner if the method is from Observer
        def call_all(items, fn):
          for item in items:
            getattr(item, fn)()
        return functools.partial(call_all, self.callbacks, attr)
      else:
        return object.__getattribute__(self, attr) # do not redirect the call


class Trainer:
  '''A class that defines training loop algirithm. Inspired by the fast.ai Learner class'''
  def __init__(self, model, train_dl, valid_dl, opt_func,
               lr, loss_func, callbacks: Iterable[Listener], train_cb=None, device='cpu'):
    self.model, self.train_dl, self.valid_dl, self.lr = model, train_dl, valid_dl, lr
    self.loss_func = loss_func
    self.opt_func = opt_func
    self.model=self.model.to(device)
    self.train_cb = TrainCallback() if train_cb is None else train_cb
    callbacks.append(self.train_cb)
    self.cbs = ListenerList(callbacks, trainer=self)
    self.device = device


  def one_batch(self, batch):
        self.batch = batch
        self.batch.to(self.device)
        self.cbs.before_batch()
        self._forward()
        if self.model.training:
            self.cbs.before_backward()
            self._backward()
            self.cbs.before_optim()
            self._optim_step()
        self.cbs.after_batch()

  def _forward(self):
    self.train_cb.forward()

  def _backward(self):
    self.train_cb.backward()

  def _optim_step(self):
    self.train_cb.optim_step()

  def one_epoch(self, train=True):
    self.model.training = train
    self.cbs.before_epoch()
    self.dl = self.train_dl if train else self.valid_dl
    for i,batch in enumerate(tqdm(self.dl, position=0, leave=True)):
      batch.id = i
      self.one_batch(batch)
    self.cbs.after_epoch()

  def before_fit(self,epochs):
      self.epochs = epochs
      self.opt = self.opt_func(self.model.parameters(), self.lr)
      self.cbs.before_fit()

  def fit(self, epochs, lr=None):
      if lr is not None: self.lr = lr
      self.before_fit(epochs)
      for e in range(epochs):
        self.epoch = e
        self.one_epoch()
        with torch.no_grad(): self.one_epoch(train=False)
      self.cbs.after_fit()


class TrainCallback(Listener):
  '''A callback that defines forward, backward and optimization steps'''
  def __init__(self, 
               trainer=None, 
               mixed_precision=False, 
               gradient_accumulation_size=None) -> None:
    self.trainer:Trainer = trainer
    self.mixed_precision = mixed_precision
    self.gradient_accumulation = gradient_accumulation_size is not None
    self.batch_size = gradient_accumulation_size
    if self.gradient_accumulation:
      self.count = 0 # count the current number of samples accumulated
  
  def before_fit(self):
    if self.mixed_precision:
      self.scaler = torch.cuda.amp.GradScaler(enabled=True)

  def forward(self):
    if self.mixed_precision:
      with torch.autocast(device_type=self.trainer.device, dtype=torch.float16):
        self.do_forward()
    else:
      self.do_forward()

  def do_forward(self):
    self.trainer.preds = self.trainer.model(self.trainer.batch)
    self.trainer.loss_grad = self.trainer.loss_func(self.trainer.preds, self.trainer.batch.y) # for gradient propagation
    self.trainer.loss = self.trainer.loss_grad.clone() # for metrics

  def backward(self):
    if self.gradient_accumulation:
      self.trainer.loss_grad /= (self.batch_size/len(self.trainer.batch.y))
    if self.mixed_precision:
      self.scaler.scale(self.trainer.loss_grad).backward()
    else:
      self.trainer.loss_grad.backward()

  def optim_step(self):
    if self.gradient_accumulation:
      self.count += len(self.trainer.batch.y)
      if self.count >= self.batch_size or self._is_last_batch():
        self.do_optim_step()
        self.count = 0
    else:
      self.do_optim_step()

  def _is_last_batch(self):
    return self.trainer.batch.id == len(self.trainer.dl) - 1
  
  def do_optim_step(self):
    optimizer = self.trainer.opt
    if self.mixed_precision:
      self.scaler.step(optimizer)
      self.scaler.update()
    else:
      optimizer.step()
    
    optimizer.zero_grad()