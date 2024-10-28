import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def train(model, dataloader, optimizer, criterion, device, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += targets.size(0)
        correct += (torch.argmax(outputs.softmax(1),1) == targets.cpu().detach()).sum()

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    # Log to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

    print(f"Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAccuracy: {accuracy:.6f}")

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
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            all_predictions.extend(outputs.softmax(1).argmax(1))
            all_targets.extend(targets.cpu().detach().numpy())

    accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    # Log to TensorBoard
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    writer.add_scalar('Precision/test', precision, epoch)
    writer.add_scalar('Recall/test', recall, epoch)
    writer.add_scalar('F1/test', f1, epoch)

    print(f"Test Epoch: {epoch} \tAccuracy: {accuracy:.6f} \tPrecision: {precision:.6f} \tRecall: {recall:.6f} \tF1: {f1:.6f}")

