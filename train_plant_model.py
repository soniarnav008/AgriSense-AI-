
import json
import random
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(
    r"C:\Users\Arnav\Downloads\PlantVillage\PlantVillage"
)

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

IMAGE_SIZE = 160
BATCH_SIZE = 4
IMAGES_PER_CLASS = 500
EPOCHS = 3
SEED = 42

CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
]

DISPLAY_LABELS = [
    "Tomato - Bacterial spot",
    "Tomato - Early blight",
    "Tomato - Healthy",
    "Tomato - Late blight",
    "Tomato - Leaf mold",
]

# ---------------- RESOURCE LIMITS ----------------

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

random.seed(SEED)
MODEL_DIR.mkdir(exist_ok=True)

# ---------------- COLLECT IMAGES ----------------

print("\nCollecting dataset...")

all_paths = []
all_labels = []

for index, class_name in enumerate(CLASSES):

    folder = DATA_DIR / class_name

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    images = list(folder.glob("*.JPG"))
    images += list(folder.glob("*.jpg"))
    images += list(folder.glob("*.jpeg"))

    random.shuffle(images)
    images = images[:IMAGES_PER_CLASS]

    print(f"{class_name}: {len(images)} images")

    if len(images) < 100:
        raise ValueError(f"Too few images in {class_name}")

    all_paths.extend(str(p) for p in images)
    all_labels.extend([index] * len(images))

# ---------------- TRAIN / VALIDATION SPLIT ----------------

train_paths = []
train_labels = []
val_paths = []
val_labels = []

for index in range(len(CLASSES)):

    indices = [
        i for i, label in enumerate(all_labels)
        if label == index
    ]

    random.shuffle(indices)
    split = int(len(indices) * 0.8)

    for i in indices[:split]:
        train_paths.append(all_paths[i])
        train_labels.append(all_labels[i])

    for i in indices[split:]:
        val_paths.append(all_paths[i])
        val_labels.append(all_labels[i])

print("\nTraining images:", len(train_paths))
print("Validation images:", len(val_paths))

# ---------------- IMAGE PIPELINE ----------------

def load_image(path, label):

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE]
    )

    image = preprocess_input(image)

    return image, label


def make_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            512,
            seed=SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=1
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


train_ds = make_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = make_dataset(
    val_paths,
    val_labels
)

# ---------------- MOBILE NET MODEL ----------------

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers for lightweight training
base_model.trainable = False

inputs = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    len(CLASSES),
    activation="softmax"
)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------

print("\nStarting lightweight training...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ---------------- SAVE MODEL ----------------

print("\nSaving model...")

model.save(MODEL_PATH)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(DISPLAY_LABELS, f, indent=4)

print("\nTRAINING COMPLETED!")
print("Model:", MODEL_PATH)
print("Labels:", LABELS_PATH)
print("Final validation accuracy:",
      round(history.history["val_accuracy"][-1] * 100, 2), "%")
