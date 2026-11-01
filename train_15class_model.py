
import json
from pathlib import Path

import tensorflow as tf

# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parent

DATASET_DIR = Path(
    r"C:\Users\Arnav\Downloads\PlantVillage\PlantVillage"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "plant_disease_model_15class.keras"
LABELS_PATH = MODEL_DIR / "labels_15class.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 4
EPOCHS = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Limit CPU thread usage
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ---------------- LOAD DATASET ----------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Total classes:", num_classes)

if num_classes != 15:
    raise ValueError(f"Expected 15 classes, found {num_classes}")

# Save labels in model output order
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

# ---------------- PREPROCESSING ----------------

def preprocess(image, label):
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = train_ds.map(
    preprocess,
    num_parallel_calls=1
).prefetch(1)

val_ds = val_ds.map(
    preprocess,
    num_parallel_calls=1
).prefetch(1)

# ---------------- BUILD MODEL ----------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(160, 160, 3))

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAINING ----------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------- SAVE MODEL ----------------

model.save(MODEL_PATH)

print("\nTRAINING COMPLETED!")
print("Model:", MODEL_PATH)
print("Labels:", LABELS_PATH)

best_accuracy = max(history.history["val_accuracy"])

print(
    "Best validation accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)
