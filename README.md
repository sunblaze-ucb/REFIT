# REFIT: a Unified Watermark Removal Framework for Deep Learning Systems with Limited Data

This repository provides a PyTorch implementation of the paper [REFIT: a Unified Watermark Removal Framework for Deep Learning Systems with Limited Data](http://arxiv.org/abs/1911.07205). It is built from the code base of [Adi et al](https://github.com/adiyoss/WatermarkNN). 

## Paper 

REFIT: a Unified Watermark Removal Framework for Deep Learning Systems with Limited Data

Xinyun Chen\*, Wenxiao Wang\*, Chris Bender, Yiming Ding, Ruoxi Jia, Bo Li, Dawn Song. (\* Equal contribution)

ACM Asia Conference on Computer and Communications Security (AsiaCCS), 2021.

## Content

The repository contains scripts for:

* watermark embedding with different type of schemes (OOD, Pattern-based, Exponential weighting and Adversarial Frontier Stitching)
* watermark removal in our REFIT framework: fine-tuning with two optional modules: Elastic Weight Consolidation (EWC) and Augmentation with Unlabeled data (AU). 

## Dependencies
Provided code runs in Python3.5 with PyTorch 1.4.0 and torchvision 0.5.0. 

It is expected to be compatible with other versions without major changes.

## Usage

### 1. Training a clean/watermarked model

One may use `train.py` to train a clean/watermarked model. 

An example that trains a **clean** model: 

```
python3 train.py --runname=cifar10_clean --save_model=cifar10_clean.t7 --dataset=cifar10 --model=resnet18
```
An example that trains a **watermarked** model:

```
python3 train.py --runname=cifar10_OOD --save_model=cifar10_OOD.t7 --dataset=cifar10 --model=resnet18 --wmtrain --wm_path=./data/trigger_set/ --wm_lbl=labels-cifar.txt
```

For more information regarding training options, please check the help message:

```
python3 train.py --help
```

#### To watermark models with different embedding schemes

Please refer to our paper for details regarding supported embedding schemes. 

Watermark sets used in this project are included in `./data`. 

* To embed OOD or pattern-based watermarks, one can simply use `--wm_path` and `--wm_lbl` to specify the corresponding watermark inputs and labels. 

* To watermark a model with exponential weighting, one needs to train a clean model with `--model=resnet18_EW` and then fine-tune it using `fine-tune.py` with `--embed` enabled and `--exp_weighing --wm2_path --wm2_lbl --wm_batch_size ` specified. An example is as follow (please refer to `python3 fine-tune.py --help` for fine-tuning options): 

  ```
  python3 train.py --runname=cifar10_clean --save_model=cifar10_clean_EW.t7 --dataset=cifar10 --model=resnet18_EW
  ```

  ```
  python3 fine-tune.py --load_path=checkpoint/cifar10_clean_EW.t7 --runname cifar10_EW --tunealllayers --dataset=cifar10 --wm_path=./data/trigger_set_EW_cifar10/ --wm_lbl=labels_flip.txt --embed --exp_weighting=2.0 --wm2_path=./data/trigger_set_EW_cifar10/ --wm2_lbl=labels_flip.txt --wm_batch_size=4
  ```

* To watermark a model with adversarial frontier stitching, one needs to train a clean model first, then generate a watermark set mixed with true adversaries and false adversaries of it using `gen_watermark_AFS.py` (which will be stored along with the clean model as a new checkpoint ended with `.afs_nowm.t7`), and finally fine-tune from checkpoint with `--wm_afs` enabled and `--wm_afs_bsize` specified. An example is as follow (please refer to `python3 gen_watermark_AFS.py --help` and `python3 fine-tune.py --help` for more information):

  ```
  python3 gen_watermark_AFS.py --load_path=checkpoint/cifar10_clean.t7 --runname=cifar10_AFS0.15 --dataset=cifar10 --model=resnet18 --eps=0.15
  ```

  ```
  python3 fine-tune.py --load_path=checkpoint/cifar10_AFS0.15.afs_nowm.t7 --runname=cifar10_AFS0.15 --dataset=cifar10 --tunealllayers --max_epochs=40 --lr 0.001 --lradj=20 --ratio=0.1 --wm_afs --wm_afs_bsize=2
  ```

  

### 2. Watermark Removal

We support in `fine-tune.py` watermark removal in our REFIT framework, fine-tuning with optional Elastic Weight Consolidation (EWC) and Augmentation with Unlabeled data (AU).

An example of simply fine-tuning is as follow:

```
python3 fine-tune.py --load_path=checkpoint/cifar10_OOD.t7 --runname cifar10_OOD_removal --tunealllayers  --max_epochs=60 --lr=0.05 --lradj=1 --dataset=cifar10_partial+0.2 --ratio=0.9 --batch_size=100
```

To enable **Elastic Weight Consolidation (EWC)**, one should specify `--EWC_coef --EWC_samples`. 

An example is as follow:

```
python3 fine-tune.py --load_path=checkpoint/cifar10_OOD.t7 --runname cifar10_OOD_removal_EWC --tunealllayers  --max_epochs=60 --lr=0.15 --lradj=1 --dataset=cifar10_partial+0.2 --ratio=0.9 --batch_size=100 --EWC_coef=10 --EWC_samples=10000
```

To enable **Augmentation with Unlabeled data (AU)**, one should specify `--extra_data --extra_data_bsize --extra_net`. 

An example is as follow:

```
python3 fine-tune.py --lr 0.1 --load_path=checkpoint/cifar10_OOD.t7 --runname cifar10_OOD_removal_AU --tunealllayers  --max_epochs=60 --lradj=15 --dataset=cifar10_partial+0.2 --ratio=0.1 --extra_data=imagenet32+0+1002+1+11 --extra_data_bsize=50 --extra_net=checkpoint/cifar10_OOD.t7
```

EWC and AU can be enabled simultaneously. 

For more information, please refer to `python3 fine-tune.py --help`.

#### Extended usage of the `--dataset` argument

In this section we will introduce the detailed usage of the `--dataset` argument with our codebase, which enables one to specify a split of a dataset for training and fine-tuning. The same formatting applies to other argument regarding datasets as well, such as `--extra_data` for AU.

- To specify CIFAR-10, CIFAR-100, the labeled split of STL-10 and the unlabeled split of STL-10, one may use `cifar10` , `cifar100`,  `stl10` and `stl10_unlabeled` respectively.
- To specify only a fraction of CIFAR-10, CIFAR-100 and the unlabeled split of STL-10, one may use `[dataset]_partial+[fraction]`. For instance, `cifar10_partial+0.2` for the first 20% of CIFAR-10 training set and `stl10_unlabeled_partial+0.5` for the first 50% of STL-10's unlabeled split.
- To specify a part of ImageNet32 (a downsampled version of ImageNet, available [here](https://patrykchrabaszcz.github.io/Imagenet32/)), one may use `imagenet32+[label_lower]+[label_upper]+[split_lower]+[split_upper]` for a split of ImageNet32, with only samples from the `[label_lower]`-th to `[label_upper] - 1`-th classes (indexed from 0 to 999) that are stored within the `[split_lower]`-th to `[split_upper] - 1`-th splits (indexed from 1 to 10) of ImageNet32. For instance, `imagenet32+0+1000+1+11` corresponds to the entire ImageNet32 and `imagenet32+500+1000+1+2` contains only samples in the first split with labels no less than 500.

#### Usage of the `--load_path_private` argument

This optional argument is used in transfer learning settings.

It specifies the path to another pre-trained model, whose linear layer will be extracted to replace the one from the current fine-tuned model when evaluating watermark accuracy.



### 3. Testing

The `predict.py` script allows you to test your model on a testing set or on a watermark set.

Examples (Please refer to `python3 predict.py --help` for more information): 

```
python3 predict.py --model_path=./checkpoint/cifar10_OOD.t7 --dataset=cifar10
```
```
python3 predict.py --model_path=./checkpoint/cifar10_OOD.t7 --wm_path=./data/trigger_set --wm_lbl=labels-cifar.txt --testwm
```

## Citation 
If you find our work useful please cite: 
```
@inproceedings{chen2021refit,
  title={REFIT: a Unified Watermark Removal Framework for Deep Learning Systems with Limited Data},
  author={Chen, Xinyun and Wang, Wenxiao and Bender, Chris and Ding, Yiming and Jia, Ruoxi and Li, Bo and Song, Dawn},
  booktitle={Proceedings of the 2021 on Asia Conference on Computer and Communications Security},
  year={2021}
}
```