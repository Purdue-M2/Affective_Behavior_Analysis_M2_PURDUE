import torch
import torch.nn as nn
from bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
# import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import precision_score, f1_score,accuracy_score, f1_score,recall_score,hamming_loss

from scipy import optimize
from dataset import AuDataset
from DFAD_model_base import AuModel

import os


checkpoint_dir = 'checkpoints'
metrics_file_path = 'traing_log.txt' 
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def threshplus_tensor(x):
    y = x.clone()
    pros = torch.nn.ReLU()
    z = pros(y)
    return z

def search_func(losses, alpha):
    return lambda x: x + (1.0/alpha)*(threshplus_tensor(losses-x).mean().item())

def searched_lamda_loss(losses, searched_lamda, alpha):
    return searched_lamda + ((1.0/alpha)*torch.mean(threshplus_tensor(losses-searched_lamda))) 


def train_epoch(model, optimizer, scheduler, criterion, train_loader,loss_type):
    model.train()
    total_loss_accumulator = 0
    all_labels = []
    all_predictions = []

    for inputs, labels in tqdm(train_loader, desc="Batch Progress"):

        inputs, labels = inputs.to(device), labels.to(device)


        enable_running_stats(model)
        output = model(inputs).squeeze() 

        if loss_type == 'erm':
            # print(labels)
            loss_ce = criterion(output, labels)
            total_loss = loss_ce
        
        if loss_type == 'dag':
            loss_ce = criterion(output, labels)
            chi_loss_np = search_func(loss_ce,alpha=0.5)
            cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
            total_loss = searched_lamda_loss(loss_ce, cutpt, alpha=0.5)
        
        total_loss.backward()
        optimizer.first_step(zero_grad=True)

        disable_running_stats(model) 
        output = model(inputs).squeeze() 

        if loss_type == 'erm':
            loss_ce = criterion(output, labels)
            total_loss = loss_ce

        if loss_type == 'dag':
            loss_ce = criterion(output, labels)
            chi_loss_np = search_func(loss_ce,alpha=0.5)
            cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
            total_loss = searched_lamda_loss(loss_ce, cutpt, alpha=0.5)
        

        total_loss.backward()
        optimizer.second_step(zero_grad=True)

        predictions = (output >= 0.5).long()
        # Accumulate labels and predictions for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())


        total_loss_accumulator += total_loss.item()

    scheduler.step()

    # Convert accumulated labels and predictions to NumPy arrays for metric computation
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Compute metrics

    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision_macro = precision_score(all_labels, all_predictions, average='macro')

    recall_macro = recall_score(all_labels, all_predictions, average='macro')

    hamming = hamming_loss(all_labels, all_predictions)  # Fraction of labels that are incorrectly predicted

    return  total_loss_accumulator / len(train_loader),  f1,precision_macro,recall_macro,hamming


def evaluate(model, criterion, val_loader,loss_type):
    model.eval()
    all_labels = []
    all_probabilities = []  # Use this to store probabilities for all samples
    total_loss_accumulator = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs).squeeze()
            probabilities = torch.sigmoid(output)  # Use sigmoid for multi-label classification

            # Collect probabilities and true labels for each batch
            all_probabilities.append(probabilities.cpu().numpy())  # Store as NumPy array
            all_labels.extend(labels.cpu().numpy())

            # Adjust the criterion for multi-label (e.g., BCEWithLogitsLoss, if not already using sigmoid in model output)
            if loss_type == 'erm':
                # print(labels)
                loss_ce = criterion(output, labels)
                total_loss = loss_ce
            
            if loss_type == 'dag':
                loss_ce = criterion(output, labels)
                chi_loss_np = search_func(loss_ce,alpha=0.5)
                cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
                total_loss = searched_lamda_loss(loss_ce, cutpt, alpha=0.5)

            total_loss_accumulator += total_loss.item()

    # Concatenate probabilities and labels from all batches
    all_probabilities = np.concatenate(all_probabilities)
    all_labels = np.array(all_labels)

    # Threshold probabilities to convert them into binary predictions
    predicted_labels = (all_probabilities >= 0.5).astype(int)

    # Calculate macro-averaged precision, recall, and F1 score
    precision_macro = precision_score(all_labels, predicted_labels, average='macro')
    recall_macro = recall_score(all_labels, predicted_labels, average='macro')
    f1_macro = f1_score(all_labels, predicted_labels, average='macro')
    hammingloss = hamming_loss(all_labels, predicted_labels)
    avg_loss = total_loss_accumulator / len(val_loader)

    return avg_loss, precision_macro, recall_macro, f1_macro, hammingloss




def model_trainer(loss_type, batch_size=64, num_epochs=32):
    # Move model to GPU
    model = AuModel().to(device)
    train_dataset = AuDataset(
    hdf5_filename='au_train.h5',
    labels_filename='au_train.txt',
    dataset_name='train_features'
)
    val_dataset = AuDataset(
    hdf5_filename='au_val.h5',
    labels_filename='au_val.txt',
    dataset_name='val_features'
)
    


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=False)


    # Prepare data loaders
    if loss_type == 'erm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    
    # Initialize optimizer and scheduler
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    # Initialize the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=num_epochs / 4, eta_min=1e-5)  # eta_min is the minimum lr


    # trian and evaluate
    with open(metrics_file_path, 'w') as metrics_file:
        for epoch in range(num_epochs):
            epoch_str = str(epoch).zfill(4)
            print(epoch_str)
            train_loss, f1,precision_macro,recall_macro,hamming = train_epoch(model, optimizer, scheduler, criterion,train_loader,loss_type)

            # val_loss, accuracy, auc= evaluate(model, criterion, val_loader)
            val_loss, precision_val, recall_val, f1_val, hammingloss= evaluate(model, criterion, val_loader,loss_type)
            

                    # Write metrics to console and file
            metrics_str = (
                f'Epoch: {epoch_str}\n'
                f'Train Loss: {train_loss:.6f}, F1: {f1:.6f}, precision_macro: {precision_macro:.6f}\n'
                f'recall_macro: {recall_macro:.6f}, F1: {f1:.6f}, hamming: {hamming:.6f}\n'
                f"val_loss: {val_loss}\n"
                f"F1: {f1_val:.6f},precision_macro: {precision_val}, recall_macro: {recall_val},hammingloss: {hammingloss}\n\n"
            )
            print(metrics_str)
            metrics_file.write(metrics_str)
            print()




            # save checkpoints
            checkpoint_name = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_name)




if __name__ == '__main__':

    model_trainer(loss_type='erm', batch_size=32, num_epochs=60)
