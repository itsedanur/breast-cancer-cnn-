import os
import shutil
import random

def split_data(source_dir, target_base, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    classes = ['benign', 'malignant']

    for cls in classes:
        src_folder = os.path.join(source_dir, cls)
        if not os.path.exists(src_folder):
            print(f"❌ Klasör bulunamadı: {src_folder}")
            continue

        images = os.listdir(src_folder)
        if not images:
            print(f"❗ {cls} klasörü boş.")
            continue

        random.shuffle(images)

        total = len(images)
        print(f"{cls.upper()} sınıfında {total} görsel bulundu.")

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        split_sets = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split_name, split_files in split_sets.items():
            dst_folder = os.path.join(target_base, split_name, cls)
            os.makedirs(dst_folder, exist_ok=True)

            for fname in split_files:
                src = os.path.join(src_folder, fname)
                dst = os.path.join(dst_folder, fname)
                shutil.copy(src, dst)

            print(f"📁 {split_name} / {cls}: {len(split_files)} görsel kopyalandı.")

    print("\n✅ Tüm veri başarıyla train/val/test olarak bölündü.")

if __name__ == "__main__":
    source_dir = "mammo_dataset"
    target_base = "data"
    split_data(source_dir, target_base)
