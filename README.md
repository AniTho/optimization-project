# Overview and Introduction

This project serves as a course project for CS 769 (Optimization in Machine learning) taught at IIT, Bombay. It involves exploring the use of submodularity for selecting subset of data to train generative adversarial network to avoid mode collapse and possible stabilise gan training via specialised loss.

## Checklist: 

- [x] Use a classifier for feature extraction which will be used for data subset selection.
- [ ] Train a W-GAN which will be used for comparison. Train it on MNIST, CIFAR, LSUN and CelebA dataset.
- [ ] Use random selection technique for selecting a small subset (say ~10-30%) of data and then train WGAN on each selected subset.
- [ ] Use submodularity for selecting a small subset (say ~10-30%) of data and then train WGAN on each selected subset.
- [ ] Check and compare the results.
- [ ] Try to incorporate submodular loss which will diminish return is similar examples are produced by the generator for penalising along with wgan loss.

## Datasets:

Add Dataset in data folder and accordingly change path in each of the dataload
- MNIST
- LSUN Bedroom (http://dl.yf.io/lsun/scenes/bedroom_train_lmdb.zip)
- Celeb A (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)