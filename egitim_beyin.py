from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def ensure_project_venv() -> None:
    project_dir = Path(__file__).resolve().parent
    if os.name == "nt":
        venv_python = project_dir / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = project_dir / ".venv" / "bin" / "python"

    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == venv_python.resolve():
        return

    print(f"Proje ortami bulundu, .venv ile yeniden baslatiliyor: {venv_python}")
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_project_venv()

try:
    import tensorflow as tf
    keras = tf.keras
    mixed_precision = tf.keras.mixed_precision
except Exception as exc:  # pragma: no cover - hard dependency
    raise ImportError(
        "TensorFlow yuklu degil veya import edilemiyor. Lutfen TensorFlow kurulumunu kontrol edin."
    ) from exc

from keras.applications import ResNet50V2
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "veriler" / "beyin"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODEL_DIR = BASE_DIR / "modeller"
MODEL_PATH = MODEL_DIR / "beyin_modeli.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 5
UNFROZEN_LAYERS = 30

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]


def validate_data_directories() -> None:
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Veri klasörleri bulunamadı. Beklenen yollar: "
            f"{TRAIN_DIR} ve {TEST_DIR}"
        )

    missing_classes = [class_name for class_name in CLASS_NAMES if not (TRAIN_DIR / class_name).exists() or not (TEST_DIR / class_name).exists()]
    if missing_classes:
        raise FileNotFoundError(
            "Aşağıdaki sınıf klasörleri hem Training hem Testing altında bulunamadı: "
            + ", ".join(missing_classes)
        )


def configure_environment() -> None:
    try:
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision etkinleştirildi: mixed_float16")
    except Exception as exc:
        print(f"Mixed precision devre dışı: {exc}")
        mixed_precision.set_global_policy("float32")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("Algılanan GPU sayısı:", len(gpus))
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def build_generators():
    train_augmenter = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    test_augmenter = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_augmenter.flow_from_directory(
        directory=str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
    )

    validation_generator = train_augmenter.flow_from_directory(
        directory=str(TRAIN_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    test_generator = test_augmenter.flow_from_directory(
        directory=str(TEST_DIR),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    print("Class indices:", train_generator.class_indices)
    return train_generator, validation_generator, test_generator


def build_model() -> Model:
    base_model = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(4, activation="softmax", dtype="float32", name="prediction")(x)

    model = Model(inputs, outputs, name="resnet50v2_beyin_tumor")
    return model


def compile_model(model: Model, learning_rate: float) -> None:
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
        ],
    )


def make_callbacks():
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def fine_tune_base_layers(model: Model, unfrozen_layers: int = UNFROZEN_LAYERS) -> None:
    base_model = next((layer for layer in model.layers if isinstance(layer, keras.Model)), None)
    if base_model is None:
        raise ValueError("Base model bulunamadi; mimariyi kontrol edin.")

    base_model.trainable = True

    for layer in base_model.layers[:-unfrozen_layers]:
        layer.trainable = False

    for layer in base_model.layers[-unfrozen_layers:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def merge_histories(*histories):
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def plot_history(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history.get("loss", [])) + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get("loss", []), label="Train Loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get("accuracy", []), label="Train Accuracy")
    if "val_accuracy" in history:
        plt.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    validate_data_directories()
    configure_environment()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_generator, validation_generator, test_generator = build_generators()

    model = build_model()

    print("\nAşama 1: Base model dondurulmuş halde eğitim başlıyor...")
    compile_model(model, learning_rate=1e-4)
    initial_history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS,
        callbacks=make_callbacks(),
        verbose=1,
    )

    print("\nAşama 2: Fine-tuning başlıyor...")
    fine_tune_base_layers(model, unfrozen_layers=UNFROZEN_LAYERS)
    compile_model(model, learning_rate=1e-5)
    fine_tune_history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=INITIAL_EPOCHS,
        callbacks=make_callbacks(),
        verbose=1,
    )

    best_model = keras.models.load_model(str(MODEL_PATH))
    metrics_summary = best_model.evaluate(test_generator, verbose=1, return_dict=True)

    print("\nTest Sonuçları:")
    for metric_name, metric_value in metrics_summary.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print(f"\nEn iyi model kaydedildi: {MODEL_PATH}")
    print(f"Sınıf sıralaması: {train_generator.class_indices}")

    history = merge_histories(initial_history, fine_tune_history)
    plot_history(history)


if __name__ == "__main__":
    main()
