# Challenge4

### 1. [download processed data](https://purdue0-my.sharepoint.com/:u:/g/personal/lin1785_purdue_edu/EeNemHHjKqhEsDDGczyKiu0BpBcI4r6IyeGeu6QgLMX0og?e=ZkpDuT)

### 2. Put the data folder under the Challenge4 directory.

### 3. Train the model
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
