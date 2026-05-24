from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="Böbrek Analiz Arayüzü",
    page_icon="🩺",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "modeller" / "bobrek_model.h5"
IMG_SIZE = (224, 224)
CLASS_LABELS = ["Cyst", "Normal", "Stone", "Tumor"]

CLASS_INFO = {
    "Cyst": "Böbrekte sıvı ile dolu iyi sınırlı yapılar. Çoğu zaman benign karakterde olup, genellikle asemptomatiktir.",
    "Normal": "Böbrek parankimi normal anatomik yapı ve yoğunluk göstermektedir; belirgin patoloji saptanmamıştır.",
    "Stone": "Böbrek taşı; mineral bileşiklerinden oluşan katı yapılar. Ağrı ve idrar retansiyonuna neden olabilir.",
    "Tumor": "Böbrekte malign veya benign neoplastik lezyon. İleri tanısal değerlendirme gerekir.",
}

RESULT_CLASSES = {
    "Cyst": "result-cyst",
    "Normal": "result-normal",
    "Stone": "result-stone",
    "Tumor": "result-tumor",
}


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    return tf.keras.models.load_model(model_path, safe_mode=False)


def is_macos_metadata_file(file_name: str) -> bool:
    return file_name.strip().startswith("._")


def load_uploaded_image(uploaded_file) -> Image.Image | None:
    if is_macos_metadata_file(uploaded_file.name):
        st.warning("._ ile başlayan metadata dosyası seçildi. Lütfen gerçek görüntü dosyasını yükleyin.")
        return None
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
        return image
    except UnidentifiedImageError:
        st.error("Dosya görüntü olarak okunamadı. Lütfen PNG/JPG/JPEG dosyası seçin.")
        return None
    except Exception as exc:
        st.error(f"Görüntü işlenirken hata oluştu: {exc}")
        return None


def predict_image(model: tf.keras.Model, uploaded_file) -> tuple[str, float]:
    uploaded_file.seek(0)
    pil_image = tf.keras.utils.load_img(BytesIO(uploaded_file.read()), target_size=IMG_SIZE)
    image_array = tf.keras.utils.img_to_array(pil_image)
    image_batch = np.expand_dims(image_array, axis=0) / 255.0

    predictions = np.asarray(model.predict(image_batch, verbose=0), dtype=np.float32).reshape(-1)
    if predictions.size != len(CLASS_LABELS):
        raise ValueError(
            f"Model çıkış boyutu ({predictions.size}) ile sınıf sayısı ({len(CLASS_LABELS)}) uyuşmuyor."
        )

    class_index = int(np.argmax(predictions))
    return CLASS_LABELS[class_index], float(predictions[class_index])


st.markdown(
    """
    <style>
    :root {
        --primary: #0a6fb6;
        --soft: #eaf4fb;
        --border: #d6e6f5;
        --ok: #118847;
        --danger: #b42318;
    }
    .app-shell {
        background: linear-gradient(135deg, #f5fbff 0%, #eef7ff 48%, #ffffff 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
    }
    .main-title {
        color: var(--primary);
        font-size: clamp(1.6rem, 2.6vw, 2.35rem);
        font-weight: 800;
        margin: 0;
    }
    .subtitle {
        color: #334155;
        margin-top: 0.45rem;
        margin-bottom: 0;
    }
    .result-box {
        border-radius: 16px;
        padding: 1rem 1.1rem;
        font-size: clamp(1.05rem, 2vw, 1.35rem);
        font-weight: 800;
        text-align: center;
        margin-top: 0.6rem;
        border: 1px solid transparent;
    }
    .result-normal { background: #e9f8ef; color: var(--ok); border-color: #b2e2c6; }
    .result-cyst { background: #ecfeff; color: #155e75; border-color: #67e8f9; }
    .result-stone { background: #fff7ed; color: #9a3412; border-color: #fdba74; }
    .result-tumor { background: #fef1f1; color: var(--danger); border-color: #fecaca; }
    .info-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        margin-top: 0.75rem;
    }
    .legal-note {
        margin-top: 2rem;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border-left: 5px solid var(--primary);
        background: var(--soft);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-shell">
        <h1 class="main-title">Yapay Zeka Destekli Böbrek Analiz Sistemi</h1>
        <p class="subtitle">Böbrek görüntülerini eğitilmiş model ile analiz eder.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model yüklenemedi: {MODEL_PATH} | Hata: {exc}")
    st.stop()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    uploaded_file = st.file_uploader(
        "PNG, JPG veya JPEG formatında böbrek BT/MR görüntüsü seçin.",
        type=["png", "jpg", "jpeg"],
    )

with right_col:
    st.subheader("Analiz Sonucu")
    if uploaded_file is None:
        st.info("Analiz için lütfen bir görüntü yükleyin.")
    else:
        image = load_uploaded_image(uploaded_file)
        if image is not None:
            st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
            try:
                result_text, confidence = predict_image(model, uploaded_file)
            except Exception as exc:
                st.error(f"Tahmin sırasında hata oluştu: {exc}")
                st.stop()

            result_class = RESULT_CLASSES.get(result_text, "result-normal")
            info_text = CLASS_INFO.get(result_text, "")

            st.markdown(f"<div class='result-box {result_class}'>{result_text}</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="info-card">
                    <b>Kısa Bilgi</b>
                    <p>{info_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            confidence_percent = int(round(confidence * 100))
            st.write(f"**Güven Skoru:** %{confidence_percent}")
            st.progress(confidence_percent)

st.markdown(
    """
    <div class="legal-note">
        Bu bir yapay zeka destekli analiz aracıdır. Kesin teşhis için uzman görüşü alınız.
    </div>
    """,
    unsafe_allow_html=True,
)
