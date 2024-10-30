import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
from torchmetrics.functional import f1_score,precision,matthews_corrcoef,accuracy
import joblib

def train(model, dataloader, optimizer, criterion, device, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probas = []
    for inputs, targets in dataloader:
        targets = targets.to(device).squeeze()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        
        probas = outputs.softmax(1) # eval probas over classes
        all_probas.append(probas)
        
        preds = torch.argmax(probas,1) # prediction argmax the probas
        all_preds.append(preds) # store preds
        all_targets.append(targets) # store targets
        
        running_loss += loss.item()
        total += targets.size(0)
        correct += (preds == targets.cpu().detach()).sum()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    all_preds   = torch.concat(all_preds  )
    all_targets = torch.concat(all_targets)
    all_probas  = torch.concat(all_probas )

    f1       = f1_score         (all_preds,all_targets,'multiclass',num_classes=2) # nc hardcoded
    prc      = precision        (all_preds,all_targets,'multiclass',num_classes=2) # nc hardcoded
    mcc      = matthews_corrcoef(all_preds,all_targets,'multiclass',num_classes=2) # nc hardcoded
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train'     , avg_loss, epoch)
    writer.add_scalar('Accuracy/train' , accuracy, epoch)
    writer.add_scalar('F1-Score/train' , f1      , epoch)
    writer.add_scalar('Precision/train', prc     , epoch)
    writer.add_scalar('MCC-Score/train', mcc     , epoch)

    for class_idx in range(2):
        writer.add_histogram(f'probabilities/class_{class_idx}', all_probas[:, class_idx], global_step=0)


    print(f"Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}")

    if epoch % 5 == 0:
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

            all_preds  .append(torch.argmax(outputs.softmax(1),1))
            all_targets.append(targets)
    
    all_preds   = torch.stack(all_preds)
    all_targets = torch.stack(all_targets)
    
    acc = accuracy         (all_preds,all_targets,'multiclass',num_classes=2)
    f1  = f1_score         (all_preds,all_targets,'multiclass',num_classes=2) # nc hardcoded
    prc = precision        (all_preds,all_targets,'multiclass',num_classes=2) # nc hardcoded
    mcc = matthews_corrcoef(all_preds,all_targets,'multiclass',num_classes=2) # nc hardcoded

    # Log to TensorBoard
    writer.add_scalar('Accuracy/test' , acc , epoch)
    writer.add_scalar('F1-Score/test' , f1  , epoch)
    writer.add_scalar('Precision/test', prc , epoch)
    writer.add_scalar('MCC-Score/test', mcc , epoch)
    

    print(f"Valid Epoch: {epoch} \tAccuracy: {acc:.6f} \tPrecision: {prc:.6f} \tF1: {f1:.6f}\tMCC: {mcc:.6f}")

