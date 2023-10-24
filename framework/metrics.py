from framework.training import Listener
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display
import pandas as pd

class LossMetric(Listener):
  '''Collects loss metrics for both train and test phases'''
  def __init__(self,trainer=None):
      if trainer is not None: self.trainer = trainer


  def before_fit(self):
    self.fit_buffer_train = []
    self.fit_buffer_eval = []

  
  def before_epoch(self):
    self.epoch_buffer = []
    self.mode = 'train' if self.trainer.model.training else 'test'

  @torch.no_grad()
  def after_batch(self):
    loss = self.trainer.loss.detach().item()
    self.epoch_buffer.append(loss)
    
  def after_epoch(self):
    avg_loss = sum(self.epoch_buffer)/len(self.epoch_buffer)
    if hasattr(self.trainer,'epoch_report'):
      self.trainer.epoch_report[f'{self.mode}_loss'] = avg_loss
    if self.mode == 'train':
      self.fit_buffer_train.extend(self.epoch_buffer) 
    else: 
      self.fit_buffer_eval.extend(self.epoch_buffer)
    

class LRMetric(Listener):
  '''Collects learning rate for each epoch'''
  def __init__(self,trainer=None):
      if trainer is not None: self.trainer = trainer

  def before_fit(self):
    self.buffer = []

  def after_epoch(self):
    if self.trainer.model.training:
      self.buffer.append(self.trainer.opt.param_groups[0]['lr'])
     

class EvalMetrics(Listener):
  '''Defines metrics collected during the evaluation step. Collects roc, auc, f1, precision, recall'''
  def __init__(self,trainer=None):
      if trainer is not None: self.trainer = trainer

  def before_fit(self):
    self.metrics = dict()

  def before_epoch(self):
    self.preds = []
    self.targs = []
    self.ids = []

  def after_batch(self):
    '''Accumulate predictions and targets'''
    if not self.trainer.model.training:
      self.preds.append(self.trainer.preds.detach())
      self.targs.append(self.trainer.batch.y.detach())
      self.ids.append(self.trainer.batch.ids) # collecting ids for the post training error analysis
  
  def get_best_precision_recall(self,precision,recall)->tuple:    
    '''Returns the best precision, recall and f1 score for a given precision and recall curve'''  
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_id = np.argmax(f1_scores)
    return precision[max_id],recall[max_id],f1_scores[max_id]


  def after_epoch(self):
    if not self.trainer.model.training:
      preds = torch.sigmoid(torch.cat(self.preds, dim=0).squeeze().detach()).cpu()
      targs = torch.cat(self.targs, dim=0).squeeze().detach().cpu()
      roc = roc_curve(targs,preds)
      precision,recall,_ = precision_recall_curve(targs,preds)
      self.precisions = precision
      self.recalls = recall
      best_precision,best_recall,best_f1 = self.get_best_precision_recall(precision,recall)
      self.metrics['roc'] = self.metrics.get('roc',[]) + [roc]
      if hasattr(self.trainer,'epoch_report'):
        self.trainer.epoch_report['auc'] = auc(*roc[:2])
        self.trainer.epoch_report['f1'] = best_f1
        self.trainer.epoch_report['recall'] = best_recall
        self.trainer.epoch_report['precision'] = best_precision
      

  def plot_roc(self):
    plt.figure()
    for i,r in enumerate(self.metrics['roc']):
      plt.plot(*r[:2], marker='*', label=i)
    plt.legend()
    plt.show()


  def plot_precision_recall(self):
    plt.figure()
    plt.plot(self.recalls,self.precisions)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()

  def get_eval_report(self):
    ids = [id for sublist in self.ids for id in sublist]
    preds = torch.sigmoid(torch.cat(self.preds, dim=0).squeeze().detach()).cpu()
    targs = torch.cat(self.targs, dim=0).squeeze().detach().cpu()
    return pd.DataFrame({'id':ids,'pred':preds,'targ':targs})

  def after_fit(self):
    if not self.trainer.model.training:
      self.plot_precision_recall()
      self.plot_roc()


class EpochReporter(Listener):
  '''Displays training stats at the end of the each epoch'''
  def __init__(self,trainer=None):
      if trainer is not None: self.trainer = trainer
      self._order = 1

  def before_fit(self):
    self.trainer.epoch_report = dict()

  def after_epoch(self):
    if not self.trainer.model.training:
      display(pd.DataFrame(self.trainer.epoch_report,index=[self.trainer.epoch]))
      self.trainer.epoch_report = dict()


