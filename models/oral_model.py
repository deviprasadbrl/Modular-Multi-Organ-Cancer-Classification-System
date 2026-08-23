import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")            

train_transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir="/kaggle/input/datasets/ashenafifasilkebede/dataset/train"
test_dir="/kaggle/input/datasets/ashenafifasilkebede/dataset/test"
val_dir="/kaggle/input/datasets/ashenafifasilkebede/dataset/val"


train_dataset=datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset=datasets.ImageFolder(root=test_dir, transform=test_transform)
val_dataset=datasets.ImageFolder(root=val_dir, transform=test_transform)

train_loader=DataLoader(dataset=train_dataset, shuffle=True, batch_size=64, num_workers=2, pin_memory=True)

test_loader=DataLoader(dataset=test_dataset, shuffle=False, batch_size=64, num_workers=2, pin_memory=True)

val_loader=DataLoader(dataset=val_dataset, shuffle=False, batch_size=64, num_workers=2, pin_memory=True)

weights=models.DenseNet121_Weights.DEFAULT
model=models.densenet121(weights=weights)

for param in model.parameters():
    param.requires_grad=False

num_fc=model.classifier.in_features
model.classifier=nn.Linear(num_fc, 2)
model=model.to(device)

optimizer_1=optim.Adam(params=model.classifier.parameters(), lr=1e-3)
criterion=nn.CrossEntropyLoss() 

print("Running Stage 1")
for epoch in range(3):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y=batch_x.to(device), batch_y.to(device)
        pred=model(batch_x)
        loss=criterion(pred, batch_y)

        optimizer_1.zero_grad()
        loss.backward()
        optimizer_1.step()


for param in model.parameters():
    param.requires_grad=True


optimizer_2 = optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])

best_val=float("inf")
patience=2
patience_counter=0

print("\nRunning Stage 2")
for epoch in range(15):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred=model(batch_x)
        loss=criterion(pred, batch_y)

        epoch_loss+=loss.item()

        optimizer_2.zero_grad()
        loss.backward()
        optimizer_2.step()

    print(f"Epoch: {epoch} loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        val_epoch_loss=0.0
        for batch_x,batch_y in val_loader:
            batch_x,batch_y=batch_x.to(device),batch_y.to(device)
            pred=model(batch_x)
            val_loss=criterion(pred,batch_y)
            val_epoch_loss+=val_loss.item()
        val_epoch_loss=val_epoch_loss/len(val_loader)
        if(val_epoch_loss<best_val):
            best_val=val_epoch_loss
            patience_counter=0
            torch.save(model.state_dict(), 'oral_expert_model.pth')
            print(f"Best model saved at epoch: {epoch}")
        else:
            patience_counter+=1
            print(f"Patience counter incresed: {patience_counter}/{patience}")
        if(patience_counter>=patience):
            print("Early Stopping Trigeered")
            break
        

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        y_pred=model(batch_x)
        y_class=torch.argmax(y_pred, dim=1)

        all_preds.extend(y_class.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

total_f1 = f1_score(all_labels, all_preds, average="macro")

print("\nModel Description: ")
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
print(f"TOTAL F1_SCORE: {total_f1:.4f}")