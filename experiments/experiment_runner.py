from dataclasses import dataclass
from torch import nn
from framework.metrics import LossMetric, LRMetric, EpochReporter, EvalMetrics
from framework.learning_rate import LrScheduler
from functools import partial
from framework.training import TrainCallback, Trainer
import torch


@dataclass
class TrainingConfig:
  device:str
  model:nn.Module
  train_dl:iter
  valid_dl:iter
  optimizer:any
  positive_class_weight:float
  lr_calculator:any
  epochs: int 
  learning_rate: float
  weight_decay: float = 0.01
  mixed_precision:bool = False
  gradient_accumulation_size:int = None
  fine_tune: bool = False
  head_pretrain_learning_rate: float = None
  head_pretrain_epochs: int = 1

    
class Training:
  def __init__(self,config:TrainingConfig):
    scheduler = partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda = config.lr_calculator)
    train_cb =  TrainCallback(mixed_precision=config.mixed_precision,
                              gradient_accumulation_size=config.gradient_accumulation_size)
    optimizer = partial(config.optimizer, weight_decay=config.weight_decay, eps=1e-5) 
    loss_function = torch.nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(config.positive_class_weight))
    self.trainer = Trainer(config.model, config.train_dl, config.valid_dl, optimizer, config.learning_rate, loss_function,
             callbacks = [LossMetric(), LRMetric(), LrScheduler(scheduler), EvalMetrics(), EpochReporter()], 
             train_cb=train_cb, device=config.device)
    self.model = config.model
    self.config = config
    

  def run(self): 
    if self.config.fine_tune:
      self.model.freeze()
      self.trainer.fit(self.config.head_pretrain_epochs, lr=self.config.head_pretrain_learning_rate)
      self.model.freeze(False)
    self.trainer.fit(self.config.epochs, lr=self.config.learning_rate)