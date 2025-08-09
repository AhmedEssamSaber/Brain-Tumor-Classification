# train.py
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from config import *
from model import CNN_TUMOR,ResNet152_TUMOR
from config import *
from utills import load_image_paths_and_labels
from preprocess import get_transforms
from eval import *

img_paths, labels = load_image_paths_and_labels(DATA_DIR)
print(f"Found {len(img_paths)} images.")

class BrainDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': torch.tensor(label, dtype=torch.long)}
    

# Load data
img_paths, labels = load_image_paths_and_labels(DATA_DIR)
transform = get_transforms()
dataset = BrainDataset(img_paths, labels, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                val_loss += criterion(outputs, labels).item()
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / total_val
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    return model, train_losses, val_losses, train_accs, val_accs, all_preds, all_labels

if __name__ == '__main__':
    sample_batch = next(iter(train_loader))
    sample_images = sample_batch['image']
    sample_input = sample_images[:1]

    # Pass shape_in param
    params["shape_in"] = tuple(sample_input.shape[1:])
    model = CNN_TUMOR(params=params)

    trained_model, train_losses, val_losses, train_accs, val_accs, y_pred, y_true = train_model(
        model, train_loader, val_loader, DEVICE, epochs=NUM_EPOCHS, lr=LEARNING_RATE
    )

    # Save results in structured folders
    loss_history = {"train_loss": train_losses, "val_loss": val_losses}
    metric_history = {"train_acc": train_accs, "val_acc": val_accs}

    save_training_plots(loss_history, metric_history, model_name="brain_tumor_model")
    evaluate_model_basic(y_true, y_pred, model_name="brain_tumor_model")
