import os
import shutil

source_dir = "PlantVillage"
healthy_dir = "dataset/train/healthy"
diseased_dir = "dataset/train/diseased"

os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(diseased_dir, exist_ok=True)

limit = 200  # max images per class

for folder in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder)

    if os.path.isdir(folder_path):
        count = 0

        for img in os.listdir(folder_path):
            if count >= limit:
                break

            src = os.path.join(folder_path, img)

            if "healthy" in folder.lower():
                dst = os.path.join(healthy_dir, img)
            else:
                dst = os.path.join(diseased_dir, img)

            try:
                shutil.copy(src, dst)
                count += 1
            except:
                pass

print("✅ Data organized successfully!")