# Robust Affective Behavior Analysis with CLIP

Li Lin, Sarah Papabathini, Xin Wang, and Shu Hu

M2-Lab@Purdue Team

This repository is the official implementation of our paper [Robust Light-Weight Facial Affective Behavior Recognition with CLIP](https://arxiv.org/abs/2403.09915).
_________________

### 1. Data Preparation
* [Data is provided by the CVPR 2024 6th ABAW Competition](https://affective-behavior-analysis-in-the-wild.github.io/6th/). 

* After getting the cropped and aligned face image data, use [CLIP ViT L/14](https://github.com/openai/CLIP) to extract image features and save them into h5 file (e.g., expr_train_clip.h5 and expr_val_clip.h5) by executing [clip_feature.py](./clip_feature.py). 
```python
python clip_feature.py
```


### 2. Train the model
#### Task Expr
* load 'expr_train_clip.h5', 'expr_train.txt' for train_dataset in [train.py](./train.py); load 'expr_val_clip.h5', 'expr_val.txt' for val_dataset in [train.py](./train.py).
```python
python train.py
```
* Use CVaR

```python
model_trainer(loss_type='dag', batch_size=32, num_epochs=32)
```

#### Task AU
* load 'expr_train.h5', 'au_train.txt' for train_dataset in [train_au.py](./train_au.py); load 'au_val.h5', 'au_val.txt' for val_dataset in [train_au.py](./train_au.py).
```python
python train_au.py
```

* Use CVaR

```python
model_trainer(loss_type='dag', batch_size=32, num_epochs=32)
```

## Citation
Please kindly consider citing our papers in your publications. 
```bash
@article{lin2024robust,
  title={Robust Light-Weight Facial Affective Behavior Recognition with CLIP},
  author={Lin, Li and Papabathini, Sarah and Wang, Xin and Hu, Shu},
  journal={arXiv preprint arXiv:2403.09915},
  year={2024}
}
```
