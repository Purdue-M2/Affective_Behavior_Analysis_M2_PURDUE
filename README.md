# Robust Affective Behavior Analysis with CLIP

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
