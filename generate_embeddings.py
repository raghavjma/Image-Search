import os
import numpy as np
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm

# === SETTINGS ===
IMAGE_FOLDER = 'images'  # folder where your images are stored
EMBEDDINGS_FILE = 'embeddings.pkl'
FILENAMES_FILE = 'filenames.pkl'

# === Load Pre-trained ResNet50 Model (without top layer) ===
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=model.input, outputs=model.output)

# === Process Images ===
filenames = []
feature_list = []

print("Extracting features...")
for img_name in tqdm(os.listdir(IMAGE_FOLDER)):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = model.predict(img_array)
        flat_features = features.flatten()
        feature_list.append(flat_features)
        filenames.append(img_path)
    except Exception as e:
        print(f"Error with image {img_path}: {e}")

# === Save to Pickle ===
with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump(feature_list, f)

with open(FILENAMES_FILE, 'wb') as f:
    pickle.dump(filenames, f)

print(f"Saved {len(feature_list)} image features to '{EMBEDDINGS_FILE}' and filenames to '{FILENAMES_FILE}'")
