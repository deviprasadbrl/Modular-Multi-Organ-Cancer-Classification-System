import os
import shutil
import random
from datasets import load_dataset
from PIL import Image

#plsce the dataset root(train/test)
source_dirs = {
    "brain": "",
    "kidney": "",
    "oral": ""
}

target_root = "./router_dataset"
train_root = os.path.join(target_root, "train")
test_root = os.path.join(target_root, "test")

samples_per_class = 400
train_split = 0.8  # 80% train, 20% test

# 2. Process Medical Domains
for domain, src_path in source_dirs.items():
    all_images = []
    for root, _, files in os.walk(src_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
                
    # Randomly select 200 total samples for this domain
    selected_images = random.sample(all_images, min(samples_per_class, len(all_images)))
    
    # Split into train and test indexes
    split_index = int(len(selected_images) * train_split)
    train_images = selected_images[:split_index]
    test_images = selected_images[split_index:]
    
    # Save to train folder
    train_dir = os.path.join(train_root, domain)
    os.makedirs(train_dir, exist_ok=True)
    for idx, img_path in enumerate(train_images):
        ext = os.path.splitext(img_path)[1]
        shutil.copy(img_path, os.path.join(train_dir, f"{domain}_{idx}{ext}"))
        
    # Save to test folder
    test_dir = os.path.join(test_root, domain)
    os.makedirs(test_dir, exist_ok=True)
    for idx, img_path in enumerate(test_images):
        ext = os.path.splitext(img_path)[1]
        shutil.copy(img_path, os.path.join(test_dir, f"{domain}_{idx}{ext}"))
        
    print(f"[{domain.upper()}] Train: {len(train_images)} | Test: {len(test_images)}")

# 3. Process 'random' class from ImageNet subset (200 total)
print("\nFetching ImageNet samples for 'random' class...")
dataset = load_dataset("Elriggs/imagenet-50-subset", split="train", streaming=True)

random_images = []
for sample in dataset:
    if len(random_images) >= samples_per_class:
        break
    img = sample['image']
    if img.mode != "RGB":
        img = img.convert("RGB")
    random_images.append(img)

split_index = int(len(random_images) * train_split)
train_rand = random_images[:split_index]
test_rand = random_images[split_index:]

# Save random train
train_rand_dir = os.path.join(train_root, "random")
os.makedirs(train_rand_dir, exist_ok=True)
for idx, img in enumerate(train_rand):
    img.save(os.path.join(train_rand_dir, f"random_{idx}.jpg"))

# Save random test
test_rand_dir = os.path.join(test_root, "random")
os.makedirs(test_rand_dir, exist_ok=True)
for idx, img in enumerate(test_rand):
    img.save(os.path.join(test_rand_dir, f"random_{idx}.jpg"))

print(f"[RANDOM] Train: {len(train_rand)} | Test: {len(test_rand)}")
print("\nDataset generation and train/test split complete!")
