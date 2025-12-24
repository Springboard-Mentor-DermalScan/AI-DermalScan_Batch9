import os
import shutil
import random

source_dir = "./Dataset"          # original dataset
target_dir = "./Dataset_split"    # new split dataset

splits = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

classes = os.listdir(source_dir)

for cls in classes:
    class_path = os.path.join(source_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * splits["train"])
    val_end = train_end + int(total * splits["val"])

    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_files.items():
        split_class_dir = os.path.join(target_dir, split, cls)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in files:
            src = os.path.join(class_path, img)
            dst = os.path.join(split_class_dir, img)
            shutil.copy(src, dst)
            
print("Dataset split completed successfully!")
