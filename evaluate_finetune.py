from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 🔹 Test verisi
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# 🔹 Modeli yüklüyorum
model = load_model('fine_tuned_model.h5')

# 🔹 Test et
loss, accuracy = model.evaluate(test_gen)
print(f"🧪 Test Loss: {loss:.4f}")
print(f"✅ Test Accuracy: {accuracy:.4f}")

# 🔹 Tahmin yap
y_pred = model.predict(test_gen)
y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
y_true = test_gen.classes

# 🔹 Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_gen.class_indices)
disp.plot(cmap=plt.cm.Purples)
plt.title("Confusion Matrix (Fine-Tuned)")
plt.show()

# 🔹 Rapor
print("\n🧾 Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=test_gen.class_indices.keys()))
input("Grafiği kapatmak için ENTER'a bas...")
