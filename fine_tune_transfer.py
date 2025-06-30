from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

# 🔹 Veriler
train_dir = 'data/train'
val_dir = 'data/val'
img_size = (224, 224)
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# 🔹 Mevcut modeli yükle
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 🔓 Fine-tuning için bazı katmanları aç
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

# 🔧 Yeni model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# 🔁 Eğit
model.fit(train_gen,
          epochs=10,
          validation_data=val_gen)

# 💾 Kaydet
model.save('fine_tuned_model.h5')
print("✅ Fine-tuned model kaydedildi: fine_tuned_model.h5")
