# PrEDAtor-200
> Pretrained Encoder-Decoder Archicture for Comma-200k

### Experiments to obtain the same accuracy for segmentation, with nearly half the parameters.

# Introduction
This is a repo containing all the various bits and pieces of code I used in my various experiments exploring self-driving datasets like [Comma-10k](https://github.com/commaai/comma10k).
     
Notably, it also links to a few modifications I made after (a ton 😉) of experimentation which pulls up in performance with the Comma10k Segmentation challenge's top score using nearly **half** the parameters.

It primarily uses a subset of the offical [Comma-2k19](https://github.com/commaai/comma2k19) dataset, called [Comma-200k](https://www.kaggle.com/datasets/neelg007/comma-200k) which I uploaded publicly on Kaggle with some basic filtering, providing blazing fast download speeds - atleast for 10% of the dataset. Kaggle's dataset size limit prevented 

# Contents
A *highly* condensed summary of my experiments - since my code is all over the place with multiple Kaggle Notebooks, Colab notebooks as well as some Git forks and moficiations I chose not to upload all of them in a spaghetti dump. For certain experiments, you're welcome to DM me ❤️

- I attempted to pre-train an unsupervised Pre-trained VQ-VAE-2 [Vector Quantization Variational autoencoder](https://arxiv.org/abs/1906.00446) on Comma-200k; It works well, but doesn't help in Segmentation due to the hierarchial encoder not propogating enough information for fine-grained tasks. It can however be used for tasks which operate over a lower rank, like regression/trajectory prediction/classification/etc.

- In one of my experiments, it turned out providing simple pre-processed images to the encoder (concatted) w/ a FPN (Fully-Pyramidal Network) decoder + HRNet (High Resolution Net preserved fine-grained features and showed promising results with lesser parameters in the current Public baseline. This was done on `256x256` image size due to resource constraints. 

| Run | Best Validation loss | Parameters | Logs | % Difference |
| --- | ----------- | --- | --- | --- |
| Comma-10k (OG) baseline | `0.0631` | ~21M | [Yassine's Base](https://pastebin.com/1zwYGG8T) | 0% |
| Comma-10k (effnet_b3) baseline | `0.0745` | ~13.2M | [Yassine's effnetb3](https://pastebin.com/peLR4svt) | -16.5% (against OG baseline) | 
| Predator-baseline | `0.0654` | ~13.3M | [Pred_HRnet](https://pastebin.com/MkP4sRA2) | **+13%** (against `effnet_b3` baseline) |

==> Giving a nearly `45.6%` decrease in parameters with a minor difference of losses - easily remedied by Hyperparameter tuning and the different selection of a seed during runs.    
<br>

<img src="https://user-images.githubusercontent.com/11617870/169167253-f18cbb8f-1c52-47eb-a23d-7d65b23acfc7.png" alt="baseline_vs_mine_plot" width="650"/>

## Reproduction details

For comparing Yassine's 'OG' runs, the code has been **MODIFIED** slightly to deal with API changes and random Colab errors which seem to plague me in particular. [This](https://github.com/neel04/predator-baseline/tree/f9b42eb23f17d8a8781dbf21fa9dda10329653ab) is the version where the `256x256` runs were done - I recommend `diff`-ing and ensuring there were no errors on my part. 

For convenience and reproducibility's sake, I forked https://github.com/neel04/predator-baseline/tree/main Yassine's repo to make things more readable and neater. Use my scripts at your own risk. 

All environments used are either Kaggle or Colab. Both of them use mostly the same underlying packages so there shouldn't be any major issues which are more than a few `pip` commands away.

### Takeway for future FSD research?

> *"Simplicity is the ultimate sophistication - Steve Jobs"*

### What's Next?

I'll be planning some future experiments on studying the vaiablilty of some *very* interesting (and 'niche') methodologies. Hope to setup another public repo and compare results in real-time 🛠️

# Credits
Thanks to Rosinality for providing such a wonderfully easy and elegant VQ-VAE-2 [implementation](https://github.com/rosinality/vq-vae-2-pytorch), Yassine Yousfi for his crystal clear baseline which I forked, and of course comma.ai for generiously providing all the datasets used. 

## Final Notes

If someone spots any bugs, mistakes or issues with this repository - please do let me know! I feel that this project could be done in a more structured way especially fully comitting to logging tools like `WandB` rather than the mess of experiments and notebooks which may introduce more headaches than help. 

PRs are more than welcome! 🤗

# Contact
Neel Gupta    
High schooler :)    
neelgupta04@outlook.com
