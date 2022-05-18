# PrEDAtor-200
> Pretrained Encoder-Decoder Archicture for Comma-200k

# Introduction
This is a repo containing all the various bits and pieces of code I used in my various experiments exploring self-driving datasets like [Comma-10k](https://github.com/commaai/comma10k).
     
Notably, it also links to a few modifications I made after (a ton 😉) of experimentation which pulls up in performance with the Comma10k Segmentation challenge's top score using nearly **half** the parameters.

It primarily uses a subset of the offical [Comma-2k19](https://github.com/commaai/comma2k19) dataset, called [Comma-200k](https://www.kaggle.com/datasets/neelg007/comma-200k) which I uploaded publicly on Kaggle with some basic filtering, providing blazing fast download speeds - atleast for 10% of the dataset. Kaggle's dataset size limit prevented 

# Contents
A *highly* condensed summary of my experiments - since my code is all over the place with multiple Kaggle Notebooks, Colab notebooks as well as some Git forks and moficiations I chose not to upload all of them in a spaghetti dump. For certain experiments, you're welcome to DM me ❤️

- I attempted to pre-train an unsupervised Pre-trained VQ-VAE-2 [Vector Quantization Variational autoencoder](https://arxiv.org/abs/1906.00446) on Comma-200k; It works well, but doesn't help in Segmentation due to the hierarchial encoder not propogating enough information for fine-grained tasks. It can however be used for tasks which operate over a lower rank, like regression/trajectory prediction/classification/etc.

- In one of my experiments, it turned out providing simple pre-processed images to the encoder (concatted) w/ a FPN (Fully-Pyramidal Network) decoder + HRNet (High Resolution Net preserved fine-grained features and showed promising results with lesser parameters in the current Public baseline. 

### Takeway for future FSD breakthroughs?

> "Simplicity is the ultimate sophistication - Steve Jobs"

### What's Next?

I'll be planning some future experiments on studying the vaiablilty of some *very* interesting (and 'niche') methodologies. Hope to setup another public repo and compare results in real-time 🛠️

# Credits
Thanks to Rosinality for providing such a wonderfully easy and elegant VQ-VAE-2 [implementation](https://github.com/rosinality/vq-vae-2-pytorch), Yassine Yousfi for his crystal clear baseline which I forked, and of course comma.ai for generiously providing all the datasets used. 

# Contact
Neel Gupta    
High schooler :)    
neelgupta04@outlook.com
