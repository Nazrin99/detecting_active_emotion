from framework.training import Listener
import math
import numpy as np

class LrScheduler(Listener):
  def __init__(self, sched, trainer=None, warmup_eps=5):
    self.sched_func = sched
    if trainer is not None: self.trainer = trainer
  def before_fit(self):
    self.sched = self.sched_func(self.trainer.opt)
  def after_epoch(self):
     if self.trainer.model.training: self.sched.step()


class CosineLRCalculator:
  def __init__(self, steps, min_lr = 1e-12, warmups = 0):
    self.steps = steps
    self.min_lr = min_lr
    self.warmups = warmups
    self.wlrs = None

  def _get_cos(self, epoch):
    return (math.cos(math.pi*(epoch/self.steps)) + 1)*0.5 + self.min_lr

  def __call__(self, epoch):
    if epoch < self.warmups:
      if self.wlrs is None: self.wlrs = np.linspace(1e-5, self._get_cos(self.warmups-1), self.warmups)
      return self.wlrs[epoch]
    return self._get_cos(epoch)
  

class ExponentialLRCalculator:
  def __init__(self, factor):
    self.factor = factor
  def __call__(self, epoch):
    if epoch == 0: return 1
    return (1 - self.factor)**epoch # every step decrease by factor %