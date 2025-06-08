# %%
import os
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("faizalkarim/flood-area-segmentation")

# access data
base_dir = Path(path)
print('No of images: ', len(os.listdir(base_dir / 'Image')))
print('No of masks: ', len(os.listdir(base_dir / 'Mask')))


# %%
from PIL import Image
import numpy as np


# %%
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32


image_dir = base_dir / 'Image'
mask_dir = base_dir / 'Mask'

def load_images_and_masks():
  images, masks = [], []
  image_files = sorted(os.listdir(image_dir))
  mask_files = sorted(os.listdir(mask_dir))

  for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, mask_file)

    # Load and resize
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT)).convert("RGB")
    mask = Image.open(mask_path).resize((IMG_WIDTH, IMG_HEIGHT)).convert("L")  # Grayscale

    # Normalize image and mask
    images.append(np.array(img) / 255.0)
    masks.append(np.array(mask) / 255.0)

  return np.array(images), np.expand_dims(np.array(masks), axis=-1)

# Load dataset
images, masks = load_images_and_masks()
print(images.shape)
print(f"Loaded {len(images)} images and masks.")


# %%
import albumentations as A
import numpy as np

augmented_images = []
augmented_masks = []

for img, msk in zip(images, masks):
  augmented_images.append(img)
  augmented_masks.append(msk)

  flipped = A.HorizontalFlip(p=1.0)(image=img, mask=msk)
  augmented_images.append(flipped['image'])
  augmented_masks.append(flipped['mask'])

  cropped = A.RandomResizedCrop(
    size=(224, 224)
  )(image=img, mask=msk)
  augmented_images.append(cropped['image'])
  augmented_masks.append(cropped['mask'])

#  randomly_erased = A.CoarseDropout(
#      hole_height_range=(0.3, 0.5),
#      hole_width_range=(0.3, 0.5),
#      fill_mask=0,
#      p=1.0
#  )(image=img, mask=msk)
#  augmented_images.append(randomly_erased['image'])
#  augmented_masks.append(randomly_erased['mask'])

  rotated = A.RandomRotate90()(image=img, mask=msk)
  augmented_images.append(rotated['image'])
  augmented_masks.append(rotated['mask'])

  brightness = A.RandomBrightnessContrast(p=1.0)(image=img, mask=msk)
  augmented_images.append(brightness['image'])
  augmented_masks.append(brightness['mask'])

# Convert lists to NumPy arrays
augmented_images = np.array(augmented_images)
augmented_masks = np.array(augmented_masks)


# %%
print("Dataset size before augmentation: ", images.shape[0])
print("Dataset size after augmentation:", augmented_images.shape[0])


# %%
import matplotlib.pyplot as plt

for img, mask in zip(augmented_images[:8], augmented_masks[:8]):
  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.title(f"Image")
  plt.imshow(img)
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.title(f"Mask")
  plt.imshow(mask, cmap='gray')
  plt.axis('off')


# %%
from utils.iou import iou_metric

# %%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split



# %%
# Train-test split
X_train, X_val, y_train, y_val = train_test_split(augmented_images, augmented_masks, test_size=0.2, random_state=42)
X_train.shape,X_val.shape

# %%
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    # Decoder
    u1 = UpSampling2D((2, 2))(c4)
    u1 = Concatenate()([u1, c3])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u1)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    u2 = UpSampling2D((2, 2))(c5)
    u2 = Concatenate()([u2, c2])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(c6)

    u3 = UpSampling2D((2, 2))(c6)
    u3 = Concatenate()([u3, c1])
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(u3)
    c7 = Conv2D(16, (3, 3), activation='relu', padding='same')(c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs, outputs)
    return model

# Create model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', iou_metric])
model.summary()

# %%
# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=BATCH_SIZE
)

# %%
model.save('flood_segmentation_model_data_augmented.keras')

# %%
# Plot training history#

# Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

# Intersection over Union
plt.subplot(1, 3, 3)
plt.plot(history.history['iou_metric'], label='Training IoU')
plt.plot(history.history['val_iou_metric'], label='Validation IoU')
plt.legend()
plt.title('IoU')
plt.show()


# %%
from tensorflow.keras.models import load_model

model = load_model('./trained_models/flood_segmentation_model_data_augmented.keras')

def visualize_predictions(num_images=5):
    preds = model.predict(X_val[:num_images])

    for i in range(num_images):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(X_val[i])

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(y_val[i].squeeze(), cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(preds[i].squeeze(), cmap='gray')

        plt.show()

visualize_predictions()

# %%
