import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
from torchmetrics.functional import f1_score,precision,matthews_corrcoef,accuracy
import joblib
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('training.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# extra code to instantiate logger, write to file

def train(model, dataloader, optimizer, criterion, device, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probas = []
    
    # inputs = subject.text
    # targets = subject.label
    for k, (inputs, targets) in enumerate(dataloader):
        targets = targets.to(device).squeeze()
        
        # Azzeramento del gradiente
        optimizer.zero_grad()
        
        # Vengono salvati gli outputs del classificatore
        outputs = model(inputs)
        
        # Viene calcolata la loss function
        loss = criterion(outputs, targets)
        
        # Back-propagation
        loss.backward()
        
        # Aggiornamento dei pesi su tutto il classificatore
        optimizer.step()
        
        # Applicazione della funzione softmax agli outputs
        # del classificatore
        probas = outputs.softmax(dim=1)
        all_probas.append(probas)
        
        #print(outputs, probas)
        
        # Viene scelta la classe con probabilità più alta
        pred_labels = torch.argmax(probas, dim=1)
        
        all_preds.append(pred_labels) # store preds
        all_targets.append(targets) # store targets
        
        #print(pred_labels.shape)
        
        logger.info(f"Loss: {loss}, k: {k}, epoch: {epoch}, preds: {pred_labels}, targets: {targets}")
        
        running_loss += loss.item()
        total += targets.size(0)
        correct += (pred_labels == targets.cpu().detach()).sum()
    
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    all_preds   = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    all_probas  = torch.concat(all_probas)
    
    f1       = f1_score(all_preds, all_targets, 'multiclass', num_classes=2) # nc hardcoded
    prc      = precision(all_preds, all_targets, 'multiclass', num_classes=2) # nc hardcoded
    mcc      = matthews_corrcoef(all_preds, all_targets, 'multiclass', num_classes=2) # nc hardcoded
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train'     , avg_loss, epoch)
    writer.add_scalar('Accuracy/train' , accuracy, epoch)
    writer.add_scalar('F1-Score/train' , f1      , epoch)
    writer.add_scalar('Precision/train', prc     , epoch)
    writer.add_scalar('MCC-Score/train', mcc     , epoch)
    
    for class_idx in range(2):
        writer.add_histogram(f'probabilities/class_{class_idx}', all_probas[:, class_idx], global_step=0)
    
    
    print(f"Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}")
    
    if epoch % 1 == 0:
        joblib.dump({
            'model'  : model,
            'loss'   : criterion,
            'optim'  : optimizer,
            'epoch'  : epoch,
            'metrics': {
                'avg_loss' : avg_loss,
                'accuracy' : accuracy,
                'f1'       : f1      ,
                'prc'      : prc     ,
                'mcc'      : mcc     ,
                'probas'   : all_probas,
            }
        },
        os.path.join(writer.log_dir,'checkpoint',f'checkpoint_epoch_{epoch}.joblib'))


#WIP
def validate(model, dataloader, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss(outputs, targets)
            running_loss += loss.item()

            predicted = torch.round(torch.sigmoid(outputs)).cpu().detach().numpy()
            all_predictions.extend(predicted)
            all_targets.extend(targets.cpu().detach().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    # Log to TensorBoard
    writer.add_scalar('Loss/validation', avg_loss, epoch)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    writer.add_scalar('Precision/validation', precision, epoch)
    writer.add_scalar('Recall/validation', recall, epoch)
    writer.add_scalar('F1/validation', f1, epoch)

    print(f"Validation Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f} \tPrecision: {precision:.6f} \tRecall: {recall:.6f} \tF1: {f1:.6f}")


def test(model, dataloader, device, epoch, writer):
    model.eval()
    all_targets = []
    all_preds   = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            targets = targets.to(device).squeeze()
            
            outputs = model(inputs)
            
            #print(outputs.shape, targets.shape)
            
            # Applicazione della funzione softmax agli outputs
            # del classificatore
            probas = outputs.softmax(dim=1)
            
            # Viene scelta la classe con probabilità più alta
            pred_labels = torch.argmax(probas, dim=1)
            
            all_preds.append(pred_labels)
            all_targets.append(targets)
    
    all_preds   = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    
    acc = accuracy(all_preds, all_targets, 'multiclass', num_classes=2)
    f1  = f1_score(all_preds, all_targets, 'multiclass', num_classes=2) # nc hardcoded
    prc = precision(all_preds, all_targets, 'multiclass', num_classes=2) # nc hardcoded
    mcc = matthews_corrcoef(all_preds, all_targets, 'multiclass', num_classes=2) # nc hardcoded
    
    # Log to TensorBoard
    writer.add_scalar('Accuracy/test' , acc , epoch)
    writer.add_scalar('F1-Score/test' , f1  , epoch)
    writer.add_scalar('Precision/test', prc , epoch)
    writer.add_scalar('MCC-Score/test', mcc , epoch)
    
    print(f"Valid Epoch: {epoch} \tAccuracy: {acc:.6f} \tPrecision: {prc:.6f} \tF1: {f1:.6f}\tMCC: {mcc:.6f}")