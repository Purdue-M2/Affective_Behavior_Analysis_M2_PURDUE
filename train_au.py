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

def calculate_L_AUC(P_scores, N_scores, gamma, p):
    P_scores = P_scores.unsqueeze(1)  # Make it a column vector
    N_scores = N_scores.unsqueeze(0)  # Make it a row vector

    # Compute the margin matrix in a vectorized form
    margin_matrix = P_scores - N_scores - gamma

    # Apply the ReLU-like condition and raise to power p
    loss_matrix = torch.where(margin_matrix < 0, (-margin_matrix) ** p, torch.zeros_like(margin_matrix))

    # Compute the final L_AUC by averaging over all elements
    L_AUC = loss_matrix.mean()

    return L_AUC

def train_epoch(model, optimizer, scheduler, criterion, train_loader,loss_type):
    model.train()
    total_loss_accumulator = 0
    all_labels = []
    all_predictions = []

    alpha_cvar = 0.5
    #------------- L_AUC parameter-------------------#
    gamma = 0.6 #(0,1]
    p = 2 # >1
    alpha = 0.8
    #------------- L_AUC parameter-------------------#


    def calculate_loss(output, labels, loss_type, criterion, compute_auc=False):
         
        loss_ce = criterion(output, labels) 
        # Directly return loss_ce for 'erm' loss type
        if loss_type == 'erm':
            return loss_ce

        # For 'dag' and 'auc' loss types, perform additional computations
        if loss_type in ['dag', 'auc']:
            chi_loss_np = search_func(loss_ce, alpha_cvar)
            cutpt = optimize.fminbound(chi_loss_np, np.min(loss_ce.cpu().detach().numpy()) - 1000.0, np.max(loss_ce.cpu().detach().numpy()))
            loss = searched_lamda_loss(loss_ce, cutpt, alpha_cvar)
            
            # If compute_auc is True and loss_type is 'auc', compute the AUC component
            if compute_auc and loss_type == 'auc':
                # positive_scores = output[labels == 1]
                # negative_scores = output[labels == 0]
                # loss_auc = calculate_L_AUC(positive_scores, negative_scores, gamma, p)
                # loss = alpha * loss + (1 - alpha) * loss_auc
                loss_auc = 0
                for i in range(labels.shape[1]):
                    positive_scores = output[:, i][labels[:, i] == 1]
                    negative_scores = output[:, i][labels[:, i] == 0]
                    if len(positive_scores) > 0 and len(negative_scores) > 0:
                        loss_auc += calculate_L_AUC(positive_scores, negative_scores, gamma, p)
                
                loss_auc /= labels.shape[1]  # Averaging over all labels
                loss = alpha * loss + (1 - alpha) * loss_auc

        return loss

    for inputs, labels in tqdm(train_loader, desc="Batch Progress"):

        inputs, labels = inputs.to(device), labels.to(device)


        enable_running_stats(model)
        output = model(inputs).squeeze()
        total_loss = calculate_loss(output, labels,loss_type,criterion, compute_auc=(loss_type == 'auc'))
        total_loss.backward()
        optimizer.first_step(zero_grad=True)

        disable_running_stats(model) 
        output = model(inputs).squeeze()
        total_loss = calculate_loss(output, labels,loss_type,criterion, compute_auc=(loss_type == 'auc')) 
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


def evaluate(model,  val_loader):
    model.eval()
    all_labels = []
    all_probabilities = []  # Use this to store probabilities for all samples


    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs).squeeze()
            probabilities = torch.sigmoid(output)  # Use sigmoid for multi-label classification

            # Collect probabilities and true labels for each batch
            all_probabilities.append(probabilities.cpu().numpy())  # Store as NumPy array
            all_labels.extend(labels.cpu().numpy())



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
 

    return  precision_macro, recall_macro, f1_macro, hammingloss




def model_trainer(loss_type, batch_size=64, num_epochs=32):
    seed = 5
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
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
    


    train_loader = DataLoader(train_dataset, batch_size=batch_size,num_workers=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,num_workers=8,shuffle=False)


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

            
            precision_val, recall_val, f1_val, hammingloss= evaluate(model, val_loader)
            

                    # Write metrics to console and file
            metrics_str = (
                f'Epoch: {epoch_str}\n'
                f'Train Loss: {train_loss:.6f}, F1: {f1:.6f}, precision_macro: {precision_macro:.6f}\n'
                f'recall_macro: {recall_macro:.6f}, F1: {f1:.6f}, hamming: {hamming:.6f}\n'
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
