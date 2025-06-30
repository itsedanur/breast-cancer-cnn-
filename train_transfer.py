import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 📁 Klasör yolları
train_dir = 'data/train'
val_dir = 'data/val'

# 📐 Parametreler
img_size = (224, 224)
batch_size = 16
epochs = 20

# 🔄 Augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# 🧠 Base Model
base_model = MobileNetV2(include_top=False, input_shape=img_size + (3,), weights='imagenet')
base_model.trainable = False  # İlk başta sadece üst katmanlar eğitilecek

# 🧱 Yeni Katmanlar
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# ⚙️ Derleme
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 🛑 Callback’ler
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('transfer_model.h5', save_best_only=True)
]

# 🏋️ Eğitim
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

# 📈 Grafik
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Transfer Learning Performansı')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid()
plt.savefig("transfer_learning_accuracy.png")
plt.show()
