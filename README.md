# ZOC
This repository is an implementation of AAAI2022 paper ["Zero-Shot Out-of-Distribution Detection Based on the Pre-trained Model CLIP"](https://arxiv.org/pdf/2109.02748.pdf)

## Datasets
cifar10, cifar100 and cifarplus are available in torchvision datasets. Tinyimagenet can be downloaded by running 

`tinyimagenet.sh`

## Training
1) Training of the Decoder_text is done once. It is used later for evaluation of all datasets. To train, please run:

`python train_decoder.py` 

## Decoder checkpoint
The fine-tuned decoder weights can be downloaded from [this Google Drive link](https://drive.google.com/file/d/1Jqz0-CW66UBNGujdvsOW00-mzMenOoJB/view?usp=sharing)

## Evaluation 
Every evaluation script loads the fine-tuned decoder from training step, please run one of the following for test results :

`python cifar10_eval.py`

`python cifar100_eval.py`

`python cifarplus_eval.py`

`python tinyimagenet_eval.py`
