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

train_imgs = set(glob.glob('./comma10k/imgs/*.png'))
val_imgs = set(glob.glob('./comma10k/imgs/*9.png'))
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
                                                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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

import_mod = importlib.import_module('.vqvae', package='Comma_VAE')

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
        self.state_dict, self.loaded = None, False #temp holders

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels = [3, 32, 32, 32, 64, 64, 64, 0]
        self._in_channels: int = 3

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 7
        self.VQVAE = import_mod.VQVAE(**self.base_args).to(self.device)
        self.newmodel = torch.nn.Sequential(*(list(self.VQVAE.children())))
        self.new_layers = get_leaf_layers(self.newmodel[:3])

    def get_stages(self):
      #return torch.nn.ModuleList([*self.newmodel[:2].to(self.device), torch.nn.Conv2d(128, 3, 2, 2).to(self.device)])
      module_list = [torch.nn.Sequential(*self.new_layers[:28]), torch.nn.Sequential(*self.new_layers[28:56]), torch.nn.Sequential(*self.new_layers[56:84]), torch.nn.Sequential(*self.new_layers[84:111]), torch.nn.Sequential(*self.new_layers[111:139]), torch.nn.Sequential(*self.new_layers[139:167]), torch.nn.Sequential(*self.new_layers[167:])] #, torch.nn.Conv2d(64, 3, 2, 2))
      [_.to(self.device) for _ in module_list]
      return torch.nn.ModuleList(module_list)

    def load_state_dict(self, state_dict):
      self.state_dict = state_dict
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
        for _, i in enumerate(stages):
          x = x.to(self.device)
          if _ == 0:
            features.append(x) #appending the first tensor as is
          x = i(x)
          print(x.shape)
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
                      'n_res_channel': 64, 'n_embed': 1024}
    },
}

out = Comma_Encoder({'in_channel': 3, 'channel': 128,
                      'n_res_block': 20,
                      'n_res_channel': 64, 'n_embed': 1024}).forward(torch.zeros((16, 3, 256, 256)))
for _ in out:
  print(f'Encode shapes: {_.shape}')

#@title Inspect Model's Seg Head { run: "auto", vertical-output: true }
run = True #@param {type:"boolean"}
if run:
  model = Unet(encoder_name="Comma_Encoder", encoder_depth=7, decoder_channels=[128,128,32,32,32,32,32], classes=256, encoder_weights='Comma200k').cuda()
  model.segmentation_head[1] = torch.nn.ConvTranspose2d(256, 6, kernel_size=(4, 4), stride=(4, 4))
  #print(model.segmentation_head)
  print(model)

model(torch.ones((16, 3, 256, 256)))

class Predator(pl.LightningModule):

    def __init__(self, learning_rate, encoder_name, encoder_depth, decoder_channels, out_classes, **kwargs):
        super().__init__()
        
        self.model = Unet(
            encoder_depth=encoder_depth, encoder_name=encoder_name, decoder_channels=decoder_channels, classes=out_classes, 
            encoder_weights='Comma200k',**kwargs
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

Predator_model = Predator(encoder_name="Comma_Encoder", encoder_depth=7,
                          decoder_channels=[64,64,64,128,128,64,64], #[64,64,64,128,128,128,64]
                          out_classes=256, learning_rate=4e-5)

mylogger = WandbLogger(project="CommaNet")

pl.seed_everything(69)

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=15,
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

# Visually inspecting model predictions 🔍"""

torch.save(trainer, './out')

dummy, _ = next(iter(validation_loader))
print(dummy.shape)

Predator_model.to('cuda')
out = Predator_model(dummy.cuda())[0, :, :, :]
print(out.shape)

out = out.argmax(dim=0)
print(out.shape)

def replace_tensor_from_dict(tensor, dictionary):
    for key, value in dictionary.items():
        tensor[tensor == key] = value
    return tensor

out_converted = replace_tensor_from_dict(out, dict(zip([0, 1, 2, 3, 4, 5], [41, 76, 90, 124, 161, 0])))

Image.fromarray(out_converted.cpu().numpy().astype(np.uint8))

trainer.predict(validation_loader, ckpt_path='./out')

# Importing model and Visualization (w/ `TorchViz` + `TensorBoard`) 🖼

import_mod = importlib.import_module('.vqvae', package='Comma_VAE')

VQVAE = import_mod.VQVAE(in_channel=3, channel=128, n_res_block=20, 
                         n_res_channel=64, n_embed=1024)

'''
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