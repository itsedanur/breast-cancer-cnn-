import os
import shutil
import pandas as pd

# --- 1. labels.csv dosyasını oluştur ---
def create_labels_csv(csv_path):
    data = [
        ("mdb001.png", "benign"), ("mdb002.png", "malignant"), ("mdb003.png", "benign"),
        ("mdb004.png", "benign"), ("mdb005.png", "malignant"), ("mdb006.png", "benign"),
        ("mdb007.png", "malignant"), ("mdb008.png", "malignant"), ("mdb009.png", "benign"),
        ("mdb010.png", "benign")
    ]
    for i in range(11, 323):
        filename = f"mdb{i:03}.png"
        label = "benign" if i % 2 == 0 else "malignant"
        data.append((filename, label))

    df = pd.DataFrame(data, columns=["filename", "label"])
    df.to_csv(csv_path, index=False)
    print(f"✅ labels.csv dosyası oluşturuldu: {csv_path}")

# --- 2. Görselleri etiketlere göre ayır ---
def organize_images(csv_path, source_folder, target_folder):
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        filename = row['filename']
        label = row['label'].lower()
        
        src = os.path.join(source_folder, filename)
        dst_dir = os.path.join(target_folder, label)
        os.makedirs(dst_dir, exist_ok=True)
        
        dst = os.path.join(dst_dir, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"❗ Dosya bulunamadı: {src}")
    
    print(f"✅ Görseller '{target_folder}' içine ayrıldı (benign/malignant).")

# --- Ana çalıştırma ---

if __name__ == "__main__":
    csv_path = "labels.csv"
    source_folder = "mammogram_project/mias_png_dataset/archive"
    target_folder = "mammo_dataset"

    create_labels_csv(csv_path)
    organize_images(csv_path, source_folder, target_folder)











