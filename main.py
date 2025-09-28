import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import kagglehub



target_dir = "/chest_xray"

# download latest version directly into target_dir
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia",
)

print("Dataset downloaded to:", path)




# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# load datasets
data_dir = path + '/chest_xray'
train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# load pre-trained model and modify
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal, Pneumonia
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


num_epochs = 5
print("Starting training...\n")
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    epoch_duration = time.time() - epoch_start
    print(f"Epoch {epoch+1} complete - Avg Loss: {running_loss/len(train_loader):.4f}, "
          f"Time: {epoch_duration:.2f}s")

total_duration = time.time() - start_time
print(f"\ntraining complete in {total_duration/60:.2f} minutes.")


print("\nstarting evaluation...")
eval_start = time.time()

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

eval_duration = time.time() - eval_start
print(f"\nEvaluation completed in {eval_duration:.2f} seconds\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_data.classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save model
torch.save(model.state_dict(), "pneumonia_detector.pth")


# Visualize Predictions
def plot_predictions(model, dataloader, class_names, device, num_images=20):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(16, 8))

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(images.size(0)):
                if images_shown == num_images:
                    plt.tight_layout()
                    plt.show()
                    return
                img = images[i].cpu().numpy().transpose((1, 2, 0))
                img = img * 0.5 + 0.5  # unnormalize
                img = np.clip(img, 0, 1)

                plt.subplot(2, num_images // 2, images_shown + 1)
                plt.imshow(img, cmap='gray')
                title_color = 'green' if preds[i] == labels[i] else 'red'
                plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}",
                          color=title_color)
                plt.axis('off')

                images_shown += 1

    plt.tight_layout()
    plt.show()

# Show predictions from test set
plot_predictions(model, test_loader, train_data.classes, device)
