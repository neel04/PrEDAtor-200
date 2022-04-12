# -*- coding: utf-8 -*-

'''
**TODO:-**
1. [x] Get ptrblack a repro
2. [x] Netron export full model to visualize
3. [ ] Try directly optimizing `CrossEntropy` + High LR
4. [ ] Transfer learning w/ Augmentations
5. [ ] Fixed leaked token

**PrEDAtor-200**

> --> `Pretrained Encoder-Decoder Archicture for Comma-200k`
'''
from pytorch_lightning.loggers import WandbLogger

import cv2
import albumentations as A
import numpy as np
import random

from tqdm.autonotebook import tqdm

import subprocess
from albumentations.core.composition import Compose
import glob
import torchvision.transforms as transforms
from vqvae import *

import glob
from PIL import Image
import importlib
import einops
import random

import matplotlib.pyplot as plt
from torchinfo import summary

from torchviz import make_dot, make_dot_from_trace
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch

"""Reading the training and validation files"""

train_imgs = set(glob.glob('/content/comma10k/imgs/*.png'))
val_imgs = set(glob.glob('/content/comma10k/imgs/*9.png'))
train_imgs = train_imgs - val_imgs

train_masks = [path.replace('imgs', 'masks') for path in train_imgs]
val_masks = [path.replace('imgs', 'masks') for path in val_imgs]

def exec_bash(command):
    
    subprocess.call(command, shell=True)

# Datasets 📜

class Comma10kDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks):
        self.imgs = list(imgs)
        self.masks = list(masks)
        self.transformer = transforms.Compose([
                                               transforms.Resize(256),
                                                transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                               ])
        self.masker = transforms.Compose([
                                               transforms.Resize(256),
                                                transforms.CenterCrop(256),
                                                transforms.ToTensor(),
                                               ])
        
        self.class_keys = [0, 1, 2, 3, 4, 5]
        self.class_values = [41, 76, 90, 124, 161, 0]
        self.class_dict = dict(zip(self.class_keys, self.class_values))

    def __len__(self):
        assert len(self.imgs) == len(self.masks)
        return len(self.imgs)
 
    def __getitem__(self, idx):
      image = cv2.imread(self.imgs[idx])
      mask = self.masker(Image.fromarray(cv2.imread(self.masks[idx], 0).astype('uint8'))) * 255

      img, mask = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), mask
      mask = np.stack([(mask == v) for v in self.class_values], axis=-1).astype('uint8')
      mask = torch.from_numpy(einops.rearrange(mask, '() h w c -> () c h w'))

      return self.transformer(img), torch.argmax(mask, dim=1)

training = Comma10kDataset(train_imgs, train_masks)
validation = Comma10kDataset(val_imgs, val_masks)

#Completing DataLoader health checks

i,m = next(iter(training))
print(f'length of training dataset: {len(training)}\nLength of validation dataset: {len(validation)}')

training_loader = DataLoader(training, 
                             batch_size=64, 
                             pin_memory=True,
                             num_workers=8,
                             prefetch_factor=2)

validation_loader = DataLoader(validation, 
                             batch_size=64, 
                             pin_memory=True,
                             num_workers=16,
                             prefetch_factor=2)

#Transfer Learning 🚀

from segmentation_models_pytorch.encoders._base import EncoderMixin
from segmentation_models_pytorch.encoders import encoders, get_preprocessing_params
from segmentation_models_pytorch import Unet, DeepLabV3, UnetPlusPlus, MAnet
from segmentation_models_pytorch.losses import *
from segmentation_models_pytorch.utils.metrics import Accuracy

#import_mod = importlib.import_module('.vqvae', package='Comma_VAE')

def get_leaf_layers(m):
    children = [*m.children()]
    if not children:
        return [m]
    leaves = []
    for l in children:
        leaves.extend(get_leaf_layers(l))
    return leaves

#t = get_leaf_layers(newmodel)

class Comma_Encoder(torch.nn.Module, EncoderMixin):

    def __init__(self, base_args, **kwargs):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_args = base_args
        try:
            self.state_dict = self.base_args['state_dict']
            self.base_args.pop('state_dict')
        except:
            pass

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels = [3, 32, 32, 32, 64, 64, 32]
        self._in_channels: int = 3

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 14
        
        if self.state_dict is not None and isinstance(self.state_dict, dict):
            print(f'{"-"*10}Loading pretrained model{"-"*10}')
            self.VQVAE = VQVAE(**self.base_args).to(self.device)
            self.VQVAE.load_state_dict(self.state_dict)
        else:
            self.VQVAE = VQVAE(**self.base_args).to(self.device)

        self.newmodel = torch.nn.Sequential(*(list(self.VQVAE.children())))
        self.new_layers = get_leaf_layers(self.newmodel[:3])

    def get_stages(self):
      #all modules for skip connections
      module_list = [torch.nn.Sequential(*self.new_layers[:12]), torch.nn.Sequential(*self.new_layers[12:26]), torch.nn.Sequential(*self.new_layers[26:38]), torch.nn.Sequential(*self.new_layers[38:50]),
                     torch.nn.Sequential(*self.new_layers[50:62]), torch.nn.Sequential(*self.new_layers[62:74]), torch.nn.Sequential(*self.new_layers[74:87]), torch.nn.Sequential(*self.new_layers[87:99]),
                     torch.nn.Sequential(*self.new_layers[99:111]), torch.nn.Sequential(*self.new_layers[111:123]), torch.nn.Sequential(*self.new_layers[123:135]), torch.nn.Sequential(*self.new_layers[135:147]),
                     torch.nn.Sequential(*self.new_layers[147:159]), torch.nn.Sequential(*self.new_layers[159:] + [torch.nn.ReLU()])]
      [_.to(self.device) for _ in module_list]
      return torch.nn.ModuleList(module_list)

    def load_state_dict(self, loaded_state_dict):
      self.base_args['state_dict'] = loaded_state_dict
      self.__init__(self.base_args)
      #self.VQVAE.load_state_dict(self.state_dict)
      #self.newmodel = torch.nn.Sequential(*(list(self.VQVAE.children())))
      #self.new_layers = get_leaf_layers(self.newmodel[:3])
      #self.loaded = True

    def forward(self, x: torch.Tensor):
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        stages = self.get_stages()
        intermediaries, features = [], []

        features.append(x)

        for _, i in enumerate(stages):
          x = x.to(self.device)
          x = i(x)
          features.append(x)

        return features    #[::-1]

encoders['Comma_Encoder'] = {
    "encoder": Comma_Encoder, # encoder class here
    "pretrained_settings": {
        "Comma200k": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": "https://filetransfer.io/data-package/aMSrvwAL/download",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        "base_args": {'in_channel': 3, 'channel': 128,
                      'n_res_block': 20,
                      'n_res_channel': 64, 'n_embed': 1024, 'state_dict':None}
    },
}

out = Comma_Encoder({'in_channel': 3, 'channel': 128,
                      'n_res_block': 20,
                      'n_res_channel': 64, 'n_embed': 1024, 'state_dict':None}).forward(torch.zeros((16, 3, 256, 256)))
for _ in out:
  print(f'Encode shapes: {_.shape}')

#@title Inspect Model's Seg Head { run: "auto", vertical-output: true }
run = True #@param {type:"boolean"}
if run:
  model = Unet(encoder_name="Comma_Encoder", encoder_depth=14, decoder_channels=[64,128,32,64,32,32,64],
               classes=256, encoder_weights='Comma200k', decoder_attention_type='scse').cuda()
               
  model.segmentation_head[1] = torch.nn.ConvTranspose2d(256, 6, kernel_size=(4, 4), stride=(4, 4)).cuda()
  
  print(model)

model(torch.ones((16, 3, 256, 256)).cuda())

class Predator(pl.LightningModule):

    def __init__(self, learning_rate, encoder_name, encoder_depth, decoder_channels, out_classes, **kwargs):
        super().__init__()
        
        self.model = Unet(
            encoder_depth=encoder_depth, encoder_name=encoder_name, decoder_channels=decoder_channels, classes=out_classes, 
            encoder_weights='Comma200k', decoder_attention_type='scse', **kwargs
        ).to(self.device)

        self.model.segmentation_head[1] = torch.nn.ConvTranspose2d(256, 6, kernel_size=(4, 4), stride=(4, 4)).to(self.device)

        self.model.encoder.requires_grad_ = True

        self.learning_rate = learning_rate

        # preprocessing parameteres for image
        params = encoders[encoder_name]['pretrained_settings']['Comma200k']
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = FocalLoss(mode='multiclass') #torch.nn.CrossEntropyLoss()
        self.val_metric = torch.nn.CrossEntropyLoss()

    def forward(self, image):
        # normalize image here
        image = image #(image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0].float()

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        tgt_mask = batch[1].squeeze(1).long() #squeezing

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1        
        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True

        loss = self.loss_fn(logits_mask, tgt_mask)

        return {
            "loss": loss
            }

    def shared_epoch_end(self, outputs, stage):
      print('\n------------Epoch End------------\n',stage)

    def training_step(self, batch, batch_idx):
        features = batch[0].float()
        masks = batch[1].squeeze(1).long()

        logits = self.forward(features)

        logs = {}

        loss = self.loss_fn(logits, masks)
        self.log('loss', loss, logger=True, prog_bar=True)
        logs["train_loss"] = loss

        return {"loss": loss, "log": logs}          

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        features = batch[0].float()
        masks = batch[1].squeeze(1).long()

        logits = self.forward(features)

        log_val_metric = self.val_metric(logits, masks)
        log_val_loss = self.loss_fn(logits, masks)

        self.log('val_CE', log_val_metric, logger=True)
        self.log('val_loss', log_val_loss, logger=True)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def configure_optimizers(self):
        myopt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #ReduceLRonPlateu scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(myopt, mode='min', factor=0.1, patience=1, verbose=True, threshold=0.01, 
                                                                threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)

        return {'optimizer': myopt, 'lr_scheduler': scheduler, "monitor": "loss"}

Predator_model = Predator(encoder_name="Comma_Encoder", encoder_depth=14,
                          decoder_channels=[64,64,64,128,128,64,64], #[64,64,64,128,128,128,64]
                          out_classes=256, learning_rate=4e-4)

mylogger = WandbLogger(project="CommaNet", name='3_Xtra_PreNorm')

pl.seed_everything(69)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=20,
    auto_lr_find=True,
    precision=16,
    auto_scale_batch_size=True,
    logger=mylogger, #pl.loggers.TensorBoardLogger("./predator_logs/"),
    weights_save_path='./',
    enable_checkpointing=False,
    check_val_every_n_epoch=2
)

trainer.fit(
    Predator_model, 
    train_dataloaders=training_loader, 
    val_dataloaders=validation_loader,
)

class UnNormalize(transforms.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

unorm = UnNormalize(**dict(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]))

# Visually inspecting model predictions 🔍"""
torch.save(Predator_model, "final_predator.ckpt")

inf_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Predator_model.to(inf_device)
Predator_model.eval()
val_iter = iter(validation_loader)

for samples in range(6):
    #move dummy to inf_device
    dummy, _ = next(val_iter)
    dummy = dummy[0].reshape(1, 3, 256, 256).to(inf_device)
    dummy.to(inf_device)

    # print which device dummy is on
    print('\n\nDUMMY:',dummy.device)
    print(dummy.shape)

    with torch.no_grad():
        out = Predator_model(dummy)[0, :, :, :]

    out = out.argmax(dim=0)

    def replace_tensor_from_dict(tensor, dictionary):
        for key, value in dictionary.items():
            tensor[tensor == key] = value
        return tensor

    out_converted = replace_tensor_from_dict(out, dict(zip([0, 1, 2, 3, 4, 5], [41, 76, 90, 124, 161, 0])))
    print(f'out_converted: {out_converted.shape}')
    sample_mask = Image.fromarray(out_converted.cpu().numpy().astype(np.uint8))

    # convert a tensor from chw to a hwc using einops
    dummy = unorm(dummy)
    dummy = einops.rearrange(dummy, '() c h w -> h w c') * 255
    #de-normalize dummy with standard deviation 0.5,0.5,0.5 and mean 0.5, 0.5, 0.5
    
    source_image = Image.fromarray(dummy.cpu().numpy().astype(np.uint8))
    #save sample mask as an image
    sample_mask.save(f'./sample_mask_{samples}.jpg')
    source_image.save(f'./source_image_{samples}.jpg')
    
    #save sample mask to wandb with mylogger
    mylogger.log_image('sample_mask', [source_image, sample_mask])

#trainer.predict(validation_loader, ckpt_path='./results')

# Importing model and Visualization (w/ `TorchViz` + `TensorBoard`) 🖼
'''
VQVAE = VQVAE(in_channel=3, channel=128, n_res_block=20, 
                         n_res_channel=64, n_embed=1024)


#Doing Model surgery.. 💉

#removing layers 3 and 6
#as they're the quantizers
newmodel = torch.nn.Sequential(*(list(VQVAE.children())))
newmodel = newmodel[:3] #first 2 encoders only

summary(Predator_model, (16, 3, 256, 256), device='cuda')

dummy = next(iter(validation_loader))[0]
#y = VQVAE(dummy)
#make_dot(y[0].mean(), params=dict(VQVAE.named_parameters()))

"""Visualizing `newmodel`"""

#@title Model to Visualize { run: "auto", vertical-output: true }
!rm -rf /content/model_viz
v_model = model #@param [newmodel, VQVAE, model] {type:"raw"}
writer = SummaryWriter('./model_viz', v_model)
writer.add_graph(v_model, dummy.cpu())
writer.close()

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir ./model_viz/
'''