# eval.py
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from config import DEVICE
import os
import json

def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(model.device)
            labels = batch['label'].float().to(model.device)

            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print("Classification Report:\n", classification_report(all_labels, all_preds, digits=4))
    return acc, f1

def find_best_threshold(model, val_loader):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(model.device)
            labels = batch['label'].float().to(model.device)

            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        preds = (np.array(all_probs) > t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"Optimal Threshold = {best_thresh:.2f} with F1 Score = {best_f1:.4f}")
    return best_thresh

def save_training_plots(loss_history, metric_history, model_name):
    base_dir = os.path.join("results", model_name)
    os.makedirs(base_dir, exist_ok=True)

    # Loss plot
    loss_dir = os.path.join(base_dir, "loss")
    os.makedirs(loss_dir, exist_ok=True)
    plt.figure()
    plt.plot(loss_history["train_loss"], label='Train Loss')
    plt.plot(loss_history["val_loss"], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(loss_dir, "loss.png"))
    plt.close()

    # Accuracy plot
    acc_dir = os.path.join(base_dir, "accuracy")
    os.makedirs(acc_dir, exist_ok=True)
    plt.figure()
    plt.plot(metric_history["train_acc"], label='Train Accuracy')
    plt.plot(metric_history["val_acc"], label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(acc_dir, "accuracy.png"))
    plt.close()



def save_history_json(history, model_name):
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "history.json")

    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)


def evaluate_model_basic(y_true, y_pred, model_name="model"):
    # Generate classification report text
    report = classification_report(y_true, y_pred, target_names=["Healthy", "Tumor"], digits=4)
    
    print("\nClassification Report:\n")
    print(report)

    # Create model-specific results folder
    model_results_dir = os.path.join("results", model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # Save classification report to a text file
    report_path = os.path.join(model_results_dir, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_dir = os.path.join(model_results_dir, "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Tumor"],
                yticklabels=["Healthy", "Tumor"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(cm_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()

    print(f"[INFO] Classification report saved to {report_path}")
    print(f"[INFO] Confusion matrix saved to {os.path.join(cm_dir, f'{model_name}_confusion_matrix.png')}")
