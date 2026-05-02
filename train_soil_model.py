"""
Train Soil Type Classifier using MobileNetV2 transfer learning.
Improved version with class weights, stronger augmentation, and better fine-tuning.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
TRAIN_DIR = "Dataset/Train"
VAL_DIR = "Dataset/test"

# Stronger augmentation to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=42
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

num_classes = train_gen.num_classes
class_names = {v: k for k, v in train_gen.class_indices.items()}
print(f"Classes: {class_names}")
print(f"Train samples: {train_gen.samples}, Val samples: {val_gen.samples}")

# Compute class weights for imbalanced data
labels = train_gen.classes
class_weights_arr = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights_arr))
print(f"Class weights: {class_weights}")

# Build model with stronger regularization
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1: Train only the top layers
print("\n--- Phase 1: Training top layers ---")
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_p1 = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

model.fit(
    train_gen, epochs=15, validation_data=val_gen,
    class_weight=class_weights, callbacks=callbacks_p1
)

# Phase 2: Fine-tune last 50 layers with very low LR
print("\n--- Phase 2: Fine-tuning (last 50 layers) ---")
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

os.makedirs("models", exist_ok=True)
callbacks_p2 = [
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7),
    ModelCheckpoint("models/soil_classifier.h5", monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

model.fit(
    train_gen, epochs=30, validation_data=val_gen,
    class_weight=class_weights, callbacks=callbacks_p2
)

# Evaluate
val_loss, val_acc = model.evaluate(val_gen)
print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")

# Save
model.save("models/soil_classifier.h5")
with open("models/soil_classes.json", "w") as f:
    json.dump(class_names, f)

print(f"\nModel saved to models/soil_classifier.h5")
print(f"Classes saved to models/soil_classes.json")
print(f"Class mapping: {class_names}")
