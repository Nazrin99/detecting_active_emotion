from torch import nn
import timm
from timm.models import VisionTransformer
from transformers import Wav2Vec2Model
import torch
from data.dataloader import Batch


class Model(nn.Module):
    '''Base class for all models experiments. Defines common methods for fine-tuning.'''
    def __init__(self):
        super().__init__()

    def freeze(self, do_freeze=True):
        '''Freeze all layers except head'''
        requires_grad = not do_freeze 
        for param in self.parameters():
            param.requires_grad = requires_grad
        if do_freeze: self.enable_head_grad()
    
    def enable_head_grad(self):
        for param in self.head.parameters():
            param.requires_grad = True


class ViTModel(Model):
    '''Timm models wrapper used for ViT models experiments'''
    def __init__(self, im_size=(64, 374), name=None, dropout=None):
        super().__init__()
        if name is None: name = 'deit_base_distilled_patch16_224'
        self.distilled = 'distilled' in name
        self.model:VisionTransformer = timm.create_model(name, img_size=im_size, pretrained=True, in_chans=1, num_classes=1)
        if dropout is not None: self.model.head_drop.p = dropout
    def forward(self,batch,latent_vector=False):
        '''forward pass with the option to return a vector representation'''
        bx = batch.x if isinstance(batch,Batch) else batch
        if latent_vector: 
            hidden = self.model.forward_features(bx)
            if self.distilled:
                return hidden[:,:2].mean(1) # take cls token, DEit has 2 cls tokens
            else:
                return hidden[:,0] # take cls token
        return self.model(bx)

    def enable_head_grad(self):
        self.model.head.requires_grad = True
        if self.distilled:
            for param in self.model.head_dist.parameters():
                param.requires_grad = True
    

class W2V2Model(Model):
    '''
    A wav2vec2 model encoder with mean pooling and a simple head on top.
    The model consumes features produced by the CNN feature extractor - Wav2Vec2FeatureEncoder 
    '''
    def __init__(self):
        super().__init__()
        w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        w2v.config.apply_spec_augment = False
        self.feature_projection = w2v.feature_projection
        self.encoder = w2v.encoder
        w2v.freeze_feature_encoder()
        self.head = nn.Sequential(
                                nn.Linear(768,256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256,64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64,1))

    def forward(self, batch,latent_vector=False):
        '''
        batch: a Batch object or a tensor. Contains features computed by Wav2Vec2FeatureEncoder
        '''
        x = batch.x if isinstance(batch,Batch) else batch
        x = self.feature_projection(x)[0]
        x = self.encoder(x)[0]
        x = x.mean((1))
        if latent_vector: return x
        return self.head(x)
    

class SimpleFusion(nn.Module):
    '''Simple fusion layer to combine w2v2 and spectrogram features'''
    def __init__(self, dim_w2v=768, dim_spctr=768) -> None:
        super(SimpleFusion,self).__init__()
        self.projection_w2v = nn.Linear(dim_w2v,128)
        self.projection_spctr = nn.Linear(dim_spctr,128)

    def forward(self,x_w2v,x_spctr):
        x_w2v = self.projection_w2v(x_w2v)
        x_spctr = self.projection_spctr(x_spctr)
        return torch.cat([x_w2v, x_spctr],dim=1)


class HybridModel(Model):
  '''A model that combines both w2v2 and spectrogram features'''
  def __init__(self, spctr_model, fusion=None):
    super().__init__()
    self.w2v2 = W2V2Model()
    self.spctr_model = spctr_model 
    self.fusion = SimpleFusion() if fusion is None else fusion
    self.head = nn.Sequential(nn.Linear(256,64),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(64,1))
    self.drop = nn.Dropout(0.2)

  def forward(self,batch):
    '''
    batch: a Batch object or a dictionary with Wav2Vec features and spectrogrmas
    '''
    x_w2v2 = self.w2v2(batch.x['w2v2'],latent_vector=True)
    x_spctr = self.spctr_model(batch.x['spctr'],latent_vector=True)
    x = self.fusion(x_w2v2, x_spctr)
    x = self.drop(x)
    return self.head(x)
  
  def enable_head_grad(self):
      super().enable_head_grad()
      for param in self.fusion.parameters():
          param.requires_grad = True


class ConvNextModel(Model):
    '''Timm models wrapper used for ConvNext experiments'''
    def __init__(self, name=None, pretrained=True, dropout=None):
        super().__init__()
        if name is None: name = 'convnext_tiny'
        self.model = timm.create_model(name, pretrained=pretrained, in_chans=1, num_classes=1)
        if dropout is not None: self.model.drop = nn.Dropout(dropout)
        
    def forward(self, batch, latent_vector=False):
        x = batch.x if isinstance(batch,Batch) else batch
        if latent_vector:
            return self.model.forward_features(x).mean((2,3)) # Global average pooling
        return self.model(x)
    
    def enable_head_grad(self):
        for param in self.model.head.parameters():
            param.requires_grad = True     


def convolution_block(input_ch, output_ch, kernel_size = 3, padding=1, act=True):
    layers = [nn.Conv2d(input_ch, output_ch, stride=1, kernel_size=kernel_size, padding=padding), nn.BatchNorm2d(output_ch)]
    if act: layers.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
  def __init__(self, input_ch, output_ch):
    super(ResidualBlock, self).__init__()
    self.noop = lambda x: x
    self.residual_conv = convolution_block(input_ch,output_ch,kernel_size=1, padding=0, act=False)
    self.residual_connection = self.noop if input_ch == output_ch else self.residual_conv
    self.conv1 = convolution_block(input_ch,output_ch)
    self.conv2 = convolution_block(output_ch,output_ch,act=False)
    self.convolutions = lambda x: self.conv2(self.conv1(x))
    self.relu = nn.LeakyReLU(0.1)
  def forward(self, x):
    return self.relu(self.convolutions(x) + self.residual_connection(x))
  

class FlattenLayer(nn.Module):
  def __init__(self): super(FlattenLayer, self).__init__()
  def forward(self, x):
    return x.view(x.size(0), -1)


class CustomConvTransformer(Model):
    '''A CNN-Transformer model that uses a CNN feature extractor and a transformer encoder on top'''
    def __init__(self, 
                 conv, 
                 conv_emb_dim, 
                 target_dim = None, 
                 layers = 3, 
                 heads=8, 
                 feedforward_x=1, 
                 dropout=0.2):
        '''
        conv: a CNN feature extractor
        conv_emb_dim: dimensionality of the CNN feature extractor output
        target_dim: dimensionality of the projected embeddings. If None, no projection is applied
        layers: number of transformer layers
        heads: number of attention heads
        feedforward_x: feedforward dimensionality multiplier
        dropout: dropout probability
        '''
        super().__init__()
        self.projection = nn.Linear(conv_emb_dim,target_dim) if target_dim is not None else nn.Identity() 
        d_model = target_dim if target_dim is not None else conv_emb_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model,d_model), nn.ReLU(), nn.Linear(d_model,1))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True, norm_first=True,
                                    dim_feedforward=d_model*feedforward_x), num_layers=layers) # B, S, D
        self.conv = conv
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self,batch):
        x = batch.x if isinstance(batch,Batch) else batch
        x = self.conv(x)
        x = x.mean(2).transpose(1,2) # average pooling on the frequency axis. Each frame is now represented by a vector
        x = self.projection(x)   # project extracted embeddings 
        x = torch.cat([self.cls.repeat(x.shape[0],1,1), x], dim=1) # add cls token
        x = self.encoder(x)
        x = x[:,0,:].squeeze(1) # take cls token
        return self.head(x)

    def enable_head_grad(self):
        super().enable_head_grad()
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True