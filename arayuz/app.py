import os
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
streamlit_dir = root_dir / ".streamlit"
config_path = streamlit_dir / "config.toml"
config_contents = """[server]
maxUploadSize = 1000
"""

try:
    os.makedirs(streamlit_dir, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(config_contents, encoding="utf-8")
    else:
        existing = config_path.read_text(encoding="utf-8")
        if "maxUploadSize" not in existing or "maxUploadSize = 1000" not in existing:
            config_path.write_text(config_contents, encoding="utf-8")
except Exception:
    pass

from contextlib import nullcontext
from datetime import datetime
import io
import tempfile

import base64
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
import tensorflow as tf
from fpdf import FPDF
from PIL import Image, UnidentifiedImageError
from streamlit_lottie import st_lottie

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except Exception:
    plt = None
    ListedColormap = None

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import torch
except Exception:
    torch = None

try:
    from model import AttentionUNet3D
except Exception:
    try:
        from karaciger_3d.model import AttentionUNet3D
    except Exception:
        AttentionUNet3D = None

st.set_page_config(
    page_title="Yapay Zeka Destekli Teşhis Sistemi",
    page_icon="🩺",
    layout="wide",
)

if "rapor_verileri" not in st.session_state:
    st.session_state["rapor_verileri"] = []
if "lab_rapor_kayitlari" not in st.session_state:
    st.session_state["lab_rapor_kayitlari"] = set()

IMG_SIZE = (224, 224)
BASE_DIR = Path(__file__).resolve().parents[1]
LUNG_MODEL_PATH = BASE_DIR / "modeller" / "akciger_modeli.h5"
BRAIN_MODEL_PATH = BASE_DIR / "modeller" / "beyin_modeli.h5"
LIVER_3D_MODEL_PATH = BASE_DIR / "karaciger_3d" / "best_model.pth"
LOGO_PATH = BASE_DIR / "arayuz" / "Proje_logo.png"
LUNG_BANNER_PATH = BASE_DIR / "arayuz" / "akciger_banner.jpeg"
BRAIN_BANNER_PATH = BASE_DIR / "arayuz" / "beyin_banner.jpeg"

BRAIN_CLASS_INFO = {
    "Glioma": "Beynin glial hücrelerinden kaynaklanan, çoğu zaman dikkatli takip ve tedavi gerektiren bir tümör tipidir.",
    "Meningioma": "Beyin zarlarından gelişen, genellikle yavaş ilerleyen ve çoğu vakada cerrahi ile değerlendirilen bir tümördür.",
    "Pituitary": "Hipofiz bezinde oluşan, hormon dengesini etkileyebilen ve endokrin değerlendirme gerektirebilen bir tümördür.",
    "Tümör Yok": "Görüntüde tümör lehine belirgin bir bulgu saptanmamıştır; yine de klinik değerlendirme önemlidir.",
}

LAB_REFERENCE_RANGES = {
    "Hemogram (Tam Kan)": {
        "WBC": {"min": 4.5, "max": 11.0, "unit": "/µL", "help": "Beyaz kan hücresi sayısı."},
        "RBC": {"min": 4.5, "max": 5.9, "unit": "m/µL", "help": "Kırmızı kan hücresi sayısı."},
        "HGB": {"min": 13.5, "max": 17.5, "unit": "g/dL", "help": "Hemoglobin düzeyi."},
        "HCT": {"min": 41.0, "max": 50.0, "unit": "%", "help": "Hematokrit yüzdesi."},
        "PLT": {"min": 150.0, "max": 450.0, "unit": "k/µL", "help": "Trombosit sayısı."},
        "MCV": {"min": 80.0, "max": 100.0, "unit": "fL", "help": "Ortalama eritrosit hacmi."},
    },
    "Biyokimya": {
        "GLU": {"min": 70.0, "max": 100.0, "unit": "mg/dL", "help": "Açlık kan şekeri."},
        "CREA": {"min": 0.7, "max": 1.3, "unit": "mg/dL", "help": "Kreatinin düzeyi."},
        "URE": {"min": 10.0, "max": 50.0, "unit": "mg/dL", "help": "Üre düzeyi."},
        "ALT": {"min": 0.0, "max": 41.0, "unit": "U/L", "help": "ALT (karaciğer enzimi)."},
        "AST": {"min": 0.0, "max": 40.0, "unit": "U/L", "help": "AST (karaciğer enzimi)."},
        "ALB": {"min": 3.5, "max": 5.0, "unit": "g/dL", "help": "Albumin düzeyi."},
    },
    "Lipid (Kolesterol)": {
        "CHOL": {"min": 0.0, "max": 200.0, "unit": "mg/dL", "help": "Toplam kolesterol."},
        "HDL": {"min": 40.0, "max": 100.0, "unit": "mg/dL", "help": "HDL (iyi kolesterol)."},
        "LDL": {"min": 0.0, "max": 130.0, "unit": "mg/dL", "help": "LDL (kötü kolesterol)."},
        "TRI": {"min": 0.0, "max": 150.0, "unit": "mg/dL", "help": "Trigliserid."},
    },
    "Vitamin & Hormon": {
        "B12": {"min": 200.0, "max": 900.0, "unit": "pg/mL", "help": "Vitamin B12 düzeyi."},
        "D-VIT": {"min": 30.0, "max": 100.0, "unit": "ng/mL", "help": "D vitamini düzeyi."},
        "FERRITIN": {"min": 20.0, "max": 250.0, "unit": "ng/mL", "help": "Ferritin düzeyi."},
        "TSH": {"min": 0.4, "max": 4.2, "unit": "mIU/L", "help": "TSH (tiroid uyarıcı hormon)."},
        "CRP": {"min": 0.0, "max": 5.0, "unit": "mg/L", "help": "CRP (iltihap göstergesi)."},
    },
}


def load_lottie_url(url: str) -> dict | None:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def render_section_header(title: str, lottie_url: str | None, height: int = 140) -> None:
    header_col, icon_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f"""
            <div class="app-shell">
                <h1 class="main-title">{title}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if lottie_url:
        animation = load_lottie_url(lottie_url)
        if animation:
            with icon_col:
                st_lottie(animation, height=height, key=f"{title}-lottie")


def render_dynamic_banner(title: str, local_path: Path | None, fallback_url: str | None) -> None:
    st.subheader(title)
    if local_path and local_path.exists():
        st.image(str(local_path), use_container_width=True)
    elif fallback_url:
        st.image(fallback_url, use_container_width=True)
    else:
        st.info("Görsel bulunamadı.")


def get_logo_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


@st.cache_resource
def load_model_from_path(model_path: str) -> tf.keras.Model:
    model_path = str(model_path)
    path_obj = Path(model_path)
    
    # Eğer sunucuda model dosyası henüz yoksa internetteki Releases kasasından çek
    if not path_obj.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Dosya adına göre ilgili GitHub Releases indirme linkini eşleştiriyoruz
        if "akciger_modeli.h5" in model_path:
            url = "https://github.com/Ernozkn/Yapay_zeka_destekli_teshis_sistemi/releases/download/v1.0/akciger_modeli.h5"
        elif "beyin_modeli.h5" in model_path:
            url = "https://github.com/Ernozkn/Yapay_zeka_destekli_teshis_sistemi/releases/download/v1.0/beyin_modeli.h5"
        else:
            raise FileNotFoundError(f"Model dosyası lokalde bulunamadı ve indirme linki eşleşmedi: {model_path}")
            
        with st.spinner(f"{path_obj.name} internetteki kasadan güvenli şekilde indiriliyor, lütfen bekleyin..."):
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                else:
                    raise RuntimeError(f"GitHub bağlantı hatası. Kod: {response.status_code}")
            except Exception as e:
                raise RuntimeError(f"Model internetten indirilirken hata oluştu: {e}")

    try:
        return tf.keras.models.load_model(model_path, safe_mode=False, compile=False, custom_objects={})
    except TypeError as exc:
        if "safe_mode" in str(exc):
            return tf.keras.models.load_model(model_path, compile=False)
        raise RuntimeError(f"Model yükleme hatası ({model_path}): {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Model yükleme hatası ({model_path}): {exc}") from exc


@st.cache_resource
def load_liver_3d_model(model_path: str, device_type: str) -> object:
    if torch is None or AttentionUNet3D is None:
        raise RuntimeError("PyTorch modeli için gerekli paketler yüklenemedi.")
    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    device = torch.device(device_type)
    model = AttentionUNet3D(
        in_channels=1,
        num_classes=3,
        base_channels=16,
        dropout=0.1,
        use_residual=True,
    ).to(device)

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(uploaded_file) -> np.ndarray:
    if isinstance(uploaded_file, Image.Image):
        pil_image = uploaded_file.convert("RGB").resize(IMG_SIZE)
    else:
        uploaded_file.seek(0)
        pil_image = tf.keras.utils.load_img(uploaded_file, target_size=IMG_SIZE)
    image_array = tf.keras.utils.img_to_array(pil_image).astype(np.float32)
    image_batch = np.expand_dims(image_array, axis=0)
    image_batch = tf.keras.applications.resnet_v2.preprocess_input(image_batch)
    return image_batch


def predict_probability(model: tf.keras.Model, image_batch: np.ndarray) -> float:
    prediction = model.predict(image_batch, verbose=0)
    probability = float(np.squeeze(prediction))
    return float(np.clip(probability, 0.0, 1.0))


def predict_class_probabilities(model: tf.keras.Model, image_batch: np.ndarray) -> np.ndarray:
    prediction = model.predict(image_batch, verbose=0)
    probabilities = np.asarray(prediction, dtype=np.float32).squeeze()
    if probabilities.ndim == 0:
        probabilities = np.array([1.0 - float(probabilities), float(probabilities)], dtype=np.float32)
    return probabilities


def is_macos_metadata_file(file_name: str) -> bool:
    return file_name.strip().startswith("._")


def load_uploaded_image(uploaded_file) -> Image.Image | None:
    if is_macos_metadata_file(uploaded_file.name):
        st.warning(
            "Seçtiğiniz dosya gerçek röntgen görüntüsü değil (. _ ile başlayan macOS metadata dosyası). "
            "Aynı klasörde ._ ile başlamayan gerçek görüntü dosyasını yükleyin."
        )
        return None

    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.load()
        return image
    except UnidentifiedImageError:
        st.error(
            "Yüklenen dosya görüntü olarak okunamadı. Lütfen geçerli bir PNG/JPG/JPEG röntgen dosyası seçin."
        )
        return None
    except Exception as exc:
        st.error(f"Görüntü işlenirken beklenmeyen bir hata oluştu: {exc}")
        return None


def load_nifti_volume(uploaded_file) -> np.ndarray:
    if nib is None:
        raise RuntimeError("nibabel yüklü değil. Lütfen nibabel kurun.")
    if uploaded_file is None:
        raise ValueError("NifTI dosyası seçilmedi.")

    suffix = ".nii.gz" if uploaded_file.name.lower().endswith(".nii.gz") else ".nii"
    uploaded_file.seek(0)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = Path(tmp_file.name)
        volume = nib.load(str(tmp_path)).get_fdata(dtype=np.float32)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"3D NifTI bekleniyordu, gelen boyut: {volume.shape}")
    return volume


def normalize_liver_volume(volume: np.ndarray) -> np.ndarray:
    hu_min = -100.0
    hu_max = 200.0
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return volume.astype(np.float32, copy=False)


def calculate_bmi(weight_kg: float, height_cm: float) -> tuple[float, str, str]:
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)

    if bmi < 18.5:
        return bmi, "Zayıf", "Beslenme düzeni ve klinik değerlendirme ile takip önerilir."
    if bmi < 25.0:
        return bmi, "Normal", "Mevcut aralık sağlıklı kabul edilir; dengeli yaşam tarzı sürdürülmelidir."
    if bmi < 30.0:
        return bmi, "Fazla Kilolu", "Kilo kontrolü için yaşam tarzı değişikliği önerilir."
    return bmi, "Obez", "Detaylı tıbbi değerlendirme ve kilo yönetimi planı önerilir."


def calculate_daily_water_need(weight_kg: float) -> float:
    return weight_kg * 0.033


def calculate_ideal_weight_range_miller(height_cm: float, gender: str) -> tuple[float, float]:
    height_in = height_cm / 2.54
    base_height_in = 60.0
    if gender == "Erkek":
        ideal = 56.2 + 1.41 * (height_in - base_height_in)
    else:
        ideal = 53.1 + 1.36 * (height_in - base_height_in)

    lower = ideal * 0.9
    upper = ideal * 1.1
    return lower, upper


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    if gender == "Erkek":
        return 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
    return 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)


def calculate_bio_age(
    age: int,
    bmi: float,
    sleep_hours: float,
    stress_level: int,
    exercise_days: int,
) -> float:
    bmi_delta = (bmi - 22.0) * 0.3
    sleep_delta = (7.0 - sleep_hours) * 0.6
    stress_delta = (stress_level - 5) * 0.7
    exercise_delta = exercise_days * -0.4
    bio_age = age + bmi_delta + sleep_delta + stress_delta + exercise_delta
    return max(10.0, bio_age)


def get_macro_distribution(goal: str) -> dict[str, int]:
    if goal == "Zayıflama":
        return {"Protein": 35, "Karbonhidrat": 35, "Yağ": 30}
    if goal == "Kas Yapma":
        return {"Protein": 30, "Karbonhidrat": 50, "Yağ": 20}
    return {"Protein": 30, "Karbonhidrat": 40, "Yağ": 30}


def get_dynamic_recommendations(
    bmi: float,
    water_needed_l: float,
    water_intake_l: float,
) -> list[tuple[str, str]]:
    recommendations: list[tuple[str, str]] = []
    if bmi >= 25.0:
        recommendations.append((
            "warning",
            "Kalp sağlığınız için kardiyo egzersizlerine odaklanın.",
        ))
    if water_intake_l < water_needed_l:
        recommendations.append((
            "success",
            "Hücre yenilenmesi için su miktarını artırın.",
        ))
    return recommendations


def footer_ekle() -> None:
    st.markdown("---")
    left_col, right_col = st.columns([3, 1])
    with left_col:
        st.caption(
            "© 2026 Karabük Üniversitesi | Yapay Zeka Operatörlüğü Mezuniyet Projesi — "
            "Sağlık profesyonelleri için karar destek asistanıdır."
        )
    with right_col:
        st.caption("📧 İletişim: ernozkn78@gmail.com")


def create_pdf_report(entries: list[dict]) -> bytes:
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    use_unicode_font = font_path.exists()
    logo_path = LOGO_PATH if LOGO_PATH.exists() else None
    watermark_path: Path | None = None

    def sanitize_text(value: str) -> str:
        if use_unicode_font:
            return value
        replacements = str.maketrans(
            "çğıöşüÇĞİÖŞÜ",
            "cgiosuCGIOSU",
        )
        return value.translate(replacements)

    class ReportPDF(FPDF):
        def __init__(self, unicode_enabled: bool, sanitizer):
            super().__init__()
            self.unicode_enabled = unicode_enabled
            self.sanitizer = sanitizer

        def footer(self):
            self.set_y(-15)
            self.set_text_color(100, 100, 100)
            if self.unicode_enabled:
                self.set_font("ArialUnicode", "", 8)
            else:
                self.set_font("Arial", "", 8)
            footer_text = "Karabük Üniversitesi - Yapay Zeka Operatörlüğü Mezuniyet Projesi"
            self.cell(0, 10, self.sanitizer(footer_text), align="C")

    def build_watermark_image(path: Path) -> Path | None:
        try:
            image = Image.open(path).convert("RGBA")
            alpha = 40
            image.putalpha(alpha)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp_path = Path(tmp_file.name)
            tmp_file.close()
            image.save(tmp_path, "PNG")
            return tmp_path
        except Exception:
            return None

    if logo_path:
        watermark_path = build_watermark_image(logo_path)

    pdf = ReportPDF(use_unicode_font, sanitize_text)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    if watermark_path:
        page_w = pdf.w
        page_h = pdf.h
        watermark_w = page_w * 0.6
        watermark_x = (page_w - watermark_w) / 2
        watermark_y = (page_h - watermark_w) / 2
        if hasattr(pdf, "set_alpha"):
            pdf.set_alpha(0.15)
        pdf.image(str(watermark_path), x=watermark_x, y=watermark_y, w=watermark_w)
        if hasattr(pdf, "set_alpha"):
            pdf.set_alpha(1)
    if logo_path:
        logo_w = 18
        pdf.image(str(logo_path), x=pdf.w - logo_w - 10, y=10, w=logo_w)
    if use_unicode_font:
        try:
            pdf.add_font("ArialUnicode", "", str(font_path), uni=True)
            pdf.set_font("ArialUnicode", "", 16)
        except Exception:
            use_unicode_font = False
            pdf.set_font("Arial", "B", 16)
    else:
        pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize_text("Klinik Analiz Raporu"), ln=True)
    pdf.ln(4)
    if use_unicode_font:
        pdf.set_font("ArialUnicode", "", 11)
    else:
        pdf.set_font("Arial", size=11)

    for item in entries:
        title = sanitize_text(str(item.get("baslik", "Kayıt")))
        if use_unicode_font:
            pdf.set_font("ArialUnicode", "", 12)
        else:
            pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, title, ln=True)
        if use_unicode_font:
            pdf.set_font("ArialUnicode", "", 11)
        else:
            pdf.set_font("Arial", size=11)
        for key, value in item.items():
            if key == "baslik":
                continue
            line = f"- {key}: {value}"
            pdf.multi_cell(0, 6, sanitize_text(line))
        pdf.ln(2)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    if watermark_path and watermark_path.exists():
        try:
            watermark_path.unlink()
        except Exception:
            pass
    return pdf_bytes


logo_data_uri = get_logo_data_uri(LOGO_PATH)
background_logo_css = ""
if logo_data_uri:
    background_logo_css = f"""
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: url('{logo_data_uri}') center center / 45% no-repeat;
        opacity: 0.08;
        pointer-events: none;
        z-index: 0;
    }}

    .stApp > div {{
        position: relative;
        z-index: 1;
    }}
    """

st.markdown(
    f"""
    <style>
    :root {{
        --medical-blue: #0a6fb6;
        --medical-blue-soft: #eaf4fb;
        --surface: #ffffff;
        --surface-strong: #f7fbff;
        --border: rgba(148, 184, 214, 0.5);
        --shadow: 0 14px 40px rgba(3, 20, 40, 0.12);
        --shadow-soft: 0 6px 22px rgba(2, 24, 46, 0.12);
        --ok: #1c7c4f;
        --danger: #8f1d1a;
        --text-main: var(--text-color, #0b1324);
        --text-muted: var(--secondary-text-color, #3c4d63);
        --glow: 0 0 0 3px rgba(32, 120, 196, 0.18);
    }}

    .stApp {{
        background: radial-gradient(circle at top, #f7fbff 0%, #eef5fb 45%, #e6eef7 100%);
        color: var(--text-main);
    }}
    {background_logo_css}

    section[data-testid="stSidebar"] > div:first-child {{
        max-height: 100vh;
        overflow-y: auto;
        padding-bottom: 1rem;
        background: linear-gradient(180deg, #f4f8fc 0%, #eef4fa 100%);
        border-right: 1px solid var(--border);
    }}

    .stButton > button {{
        border-radius: 16px;
        padding: 0.55rem 1.2rem;
        border: 1px solid rgba(15, 106, 175, 0.35);
        background: linear-gradient(135deg, #0f6eb6 0%, #1386d9 100%);
        color: #ffffff;
        box-shadow: var(--shadow-soft);
        transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 12px 30px rgba(12, 102, 170, 0.28);
        filter: brightness(1.05);
    }}

    .stButton > button:focus {{
        outline: none;
        box-shadow: var(--shadow-soft), var(--glow);
    }}

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox [data-baseweb="select"] > div,
    .stTextArea textarea {{
        border-radius: 16px;
        border: 1px solid var(--border) !important;
        background: #ffffff;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.08);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }}

    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox [data-baseweb="select"] > div:focus-within,
    .stTextArea textarea:focus {{
        border-color: rgba(12, 98, 164, 0.6) !important;
        box-shadow: var(--glow);
    }}

    .stSlider [data-baseweb="slider"] > div {{
        border-radius: 999px;
    }}

    .stSlider [data-baseweb="slider"] [role="slider"] {{
        box-shadow: 0 6px 18px rgba(12, 106, 176, 0.25);
        border: 2px solid #ffffff;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-main);
    }}

    [data-testid="stMetricLabel"] {{
        font-weight: 600;
        color: var(--text-muted);
    }}

    [data-testid="stMetricDelta"] {{
        color: var(--ok);
        font-weight: 700;
    }}

    [data-testid="stMetricDelta"] svg {{
        color: var(--ok);
    }}

    .stMetric {{
        background: var(--surface-strong);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: var(--shadow-soft);
    }}

    .stProgress > div > div {{
        background: linear-gradient(90deg, rgba(10, 111, 182, 0.2), #0a6fb6, rgba(10, 111, 182, 0.2));
        box-shadow: 0 0 12px rgba(10, 111, 182, 0.5);
    }}

    .scan-box {{
        position: relative;
        height: 120px;
        border-radius: 16px;
        border: 1px dashed rgba(10, 111, 182, 0.45);
        background: linear-gradient(180deg, rgba(10, 111, 182, 0.12), rgba(10, 111, 182, 0.02));
        overflow: hidden;
        margin-top: 0.5rem;
        box-shadow: inset 0 0 25px rgba(10, 111, 182, 0.08);
    }}

    .scan-line {{
        position: absolute;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, rgba(10, 111, 182, 0.2), #0a6fb6, rgba(10, 111, 182, 0.2));
        animation: scan 1.6s ease-in-out infinite;
        box-shadow: 0 0 14px rgba(10, 111, 182, 0.5);
    }}

    @keyframes scan {{
        0% {{ top: 8px; opacity: 0.2; }}
        50% {{ top: 88px; opacity: 0.95; }}
        100% {{ top: 8px; opacity: 0.2; }}
    }}

    .app-shell {{
        background: linear-gradient(135deg, #ffffff 0%, #f1f7fe 55%, #eaf2fb 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
    }}

    .main-title {{
        color: var(--medical-blue);
        font-size: clamp(1.6rem, 2.6vw, 2.35rem);
        font-weight: 800;
        line-height: 1.2;
        margin: 0;
    }}

    .subtitle {{
        color: var(--text-muted);
        font-size: 1rem;
        margin-top: 0.45rem;
        margin-bottom: 0;
    }}

    .result-box {{
        border-radius: 18px;
        padding: 1rem 1.1rem;
        font-size: clamp(1.05rem, 2vw, 1.35rem);
        font-weight: 800;
        text-align: center;
        margin-top: 0.6rem;
        border: 1px solid transparent;
        box-shadow: var(--shadow-soft);
    }}

    .result-normal {{
        background: #e9f8ef;
        color: var(--ok);
        border-color: #b2e2c6;
    }}

    .result-pneumonia {{
        background: #fef1f1;
        color: var(--danger);
        border-color: #fecaca;
    }}

    .result-brain {{
        background: linear-gradient(135deg, #eef7ff 0%, #f3fbff 100%);
        color: #084c8d;
        border-color: #bfdcff;
    }}

    .result-glioma {{
        background: #fff1f2;
        color: #9f1239;
        border-color: #fda4af;
    }}

    .result-meningioma {{
        background: #fff7ed;
        color: #9a3412;
        border-color: #fdba74;
    }}

    .result-pituitary {{
        background: #ecfeff;
        color: #155e75;
        border-color: #67e8f9;
    }}

    .result-notumor {{
        background: #ecfdf5;
        color: #166534;
        border-color: #86efac;
    }}

    .info-card {{
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        box-shadow: var(--shadow-soft);
        margin-top: 0.75rem;
    }}

    .info-title {{
        color: var(--medical-blue);
        font-weight: 800;
        margin-bottom: 0.35rem;
    }}

    .info-text {{
        color: var(--text-main);
        margin: 0;
        line-height: 1.55;
    }}

    .legal-note {{
        margin-top: 2rem;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border-left: 5px solid var(--medical-blue);
        background: var(--medical-blue-soft);
        color: var(--text-main);
        font-size: 0.95rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-shell">
        <h1 class="main-title">Yapay Zeka Destekli Teşhis Sistemi</h1>
        <p class="subtitle">Akciğer ve beyin görüntülerini eğitilmiş derin öğrenme modelleri ile analiz eder.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

menu_selection = st.sidebar.radio(
    "Menü",
    [
        "Ana Sayfa 🏠",
        "Akciğer Analizi 🫁",
        "Beyin MR Analizi 🧠",
        "Karaciğer 3D Analizi 🚀",
        "Vücut Analizi (BMI/BMR) 🏋️‍♂️",
        "Laboratuvar Analiz Rehberi 🧪",
        "Sağlıklı Yaşam Rehberi 🍎",
        "Sağlık Raporu 📄",
        "Bilgi Bankası 📗",
        "Destek ve İletişim 📞",
        "Hakkında ℹ️",
    ],
    index=0,
)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
else:
    st.sidebar.warning("Logo bulunamadı: Proje_logo.png")
st.sidebar.caption("AI Health Assistant v1.0")

lottie_animation = load_lottie_url(
    "https://assets6.lottiefiles.com/packages/lf20_9cyyl8i4.json"
)
if lottie_animation:
    st.sidebar.markdown("### Sağlık Animasyonu")
    st_lottie(lottie_animation, height=160, key="health-lottie")

st.sidebar.markdown(
    """
    <div class="info-card">
        <div class="info-title">Kullanım</div>
        <p class="info-text">Sol menüden analiz türünü seçin, ardından uygun görüntü dosyasını yükleyin.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def render_home_section() -> None:
    render_section_header(
        "Yapay Zeka Destekli Sağlık Teşhis ve Analiz Sistemi",
        "https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json",
        height=180,
    )
    st.markdown(
        """
        <p class="subtitle">Derin öğrenme ile medikal görüntü analizi ve kişisel sağlık hesaplamaları.</p>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown(
                """
                ### Sistem Özellikleri
                - 🧪 **Görüntü İşleme:** Akciğer ve beyin MR görüntülerinde otomatik analiz.
                - 📊 **BMI Analizi:** Vücut kütle indeksi ve su ihtiyacı hesaplama.
                - 🔥 **BMR Hesaplama:** Bazal metabolizma ve günlük kalori ihtiyacı.
                """
            )
        with right:
            st.markdown(
                """
                ### Kullanılan Model
                - **ResNet50V2** tabanlı derin öğrenme mimarisi
                - Hızlı ve güvenilir sınıflandırma
                """
            )

    st.markdown("### Nasıl Çalışır?")
    step_col_1, step_col_2, step_col_3 = st.columns(3)
    with step_col_1:
        st.markdown("**1. Görüntü yükle**")
        st.caption("Akciğer röntgeni veya beyin MR görüntüsü seçilir.")
    with step_col_2:
        st.markdown("**2. Model analizi**")
        st.caption("Model, görüntüyü otomatik olarak analiz eder.")
    with step_col_3:
        st.markdown("**3. Sonuç raporu**")
        st.caption("Tahmin ve güven skoru kullanıcıya sunulur.")
    footer_ekle()


def render_lung_section() -> None:
    page_title = "Yapay Zeka Destekli Akciğer Analiz Sistemi"
    header_col, image_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f"""
            <div class="app-shell">
                <h1 class="main-title">{page_title}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with image_col:
        if LUNG_BANNER_PATH.exists():
            st.image(str(LUNG_BANNER_PATH), width=300)
        else:
            st.caption("akciger_banner.png bulunamadı.")
    st.markdown(
        """
        <p class="subtitle">Seçilen analiz türüne uygun model ile yüklenen görüntü değerlendirilir.</p>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model_from_path(str(LUNG_MODEL_PATH))
    except Exception as exc:
        st.error(f"Model yüklenemedi: {LUNG_MODEL_PATH} | Hata: {exc}")
        st.stop()

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Görüntü Yükle")
        uploaded_file = st.file_uploader(
            "PNG, JPG veya JPEG formatında akciğer röntgeni seçin.",
            type=["png", "jpg", "jpeg"],
            help="Dosyayı sürükleyip bırakabilir veya tıklayarak seçebilirsiniz.",
        )

    with right_col:
        st.subheader("Analiz Sonucu")

        if uploaded_file is None:
            st.info("Analiz için lütfen bir görüntü yükleyin.")
        else:
            image = load_uploaded_image(uploaded_file)
            if image is not None:
                st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
                scan_placeholder = st.empty()
                scan_placeholder.markdown(
                    """
                    <div>
                        <strong>Röntgen taranıyor...</strong>
                        <div class="scan-box"><div class="scan-line"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                input_batch = preprocess_image(image)
                pneumonia_probability = predict_probability(model, input_batch)
                scan_placeholder.empty()
                if pneumonia_probability > 0.5:
                    result_text = "ZATÜRRE (PNEUMONIA) BELİRTİSİ SAPTANDI"
                    result_class = "result-pneumonia"
                    confidence = pneumonia_probability
                    info_text = "Akciğer parankiminde enfeksiyon lehine opasiteler veya infiltrasyonlar görülebilir."
                else:
                    result_text = "SAĞLIKLI (NORMAL) GÖRÜNTÜ"
                    result_class = "result-normal"
                    confidence = 1.0 - pneumonia_probability
                    info_text = "Belirgin pnömoni bulgusu saptanmadı; klinik durumla birlikte değerlendirilmelidir."

                st.markdown(
                    f"<div class='result-box {result_class}'>{result_text}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-title">Kısa Bilgi</div>
                        <p class="info-text">{info_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                confidence_percent = int(round(confidence * 100))
                st.write(f"**Güven Skoru:** %{confidence_percent}")
                st.progress(confidence_percent)
            else:
                st.info("Analiz için geçerli bir görüntü yükleyin.")

        if uploaded_file is not None:
            if st.button("Analiz Sonucunu Rapora Kaydet", key="lung-report"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state["rapor_verileri"].append(
                    {
                        "baslik": "Akciğer Analizi",
                        "Tarih": timestamp,
                        "Sonuç": result_text if image is not None else "Görüntü okunamadı",
                        "Güven Skoru": f"%{confidence_percent}" if image is not None else "-",
                    }
                )
                st.success("Rapor kaydedildi.")
    footer_ekle()


def render_brain_section() -> None:
    page_title = "Yapay Zeka Destekli Beyin MR Analiz Sistemi"
    header_col, image_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            f"""
            <div class="app-shell">
                <h1 class="main-title">{page_title}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with image_col:
        if BRAIN_BANNER_PATH.exists():
            st.image(str(BRAIN_BANNER_PATH), width=300)
        else:
            st.caption("beyin_banner.png bulunamadı.")
    st.markdown(
        """
        <p class="subtitle">Seçilen analiz türüne uygun model ile yüklenen görüntü değerlendirilir.</p>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model_from_path(str(BRAIN_MODEL_PATH))
    except Exception as exc:
        st.error(f"Model yüklenemedi: {BRAIN_MODEL_PATH} | Hata: {exc}")
        st.stop()

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("Görüntü Yükle")
        uploaded_file = st.file_uploader(
            "PNG, JPG veya JPEG formatında beyin MR görüntüsü seçin.",
            type=["png", "jpg", "jpeg"],
            help="Dosyayı sürükleyip bırakabilir veya tıklayarak seçebilirsiniz.",
        )

    with right_col:
        st.subheader("Analiz Sonucu")

        if uploaded_file is None:
            st.info("Analiz için lütfen bir görüntü yükleyin.")
        else:
            image = load_uploaded_image(uploaded_file)
            if image is not None:
                st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
                with st.spinner("Nöronlar analiz ediliyor..."):
                    input_batch = preprocess_image(image)
                    probabilities = predict_class_probabilities(model, input_batch)
                class_index = int(np.argmax(probabilities))
                brain_class_names = ["Glioma", "Meningioma", "Tümör Yok", "Pituitary"]
                brain_result_classes = {
                    "Glioma": "result-glioma",
                    "Meningioma": "result-meningioma",
                    "Pituitary": "result-pituitary",
                    "Tümör Yok": "result-notumor",
                }
                result_text = brain_class_names[class_index]
                result_class = brain_result_classes.get(result_text, "result-brain")
                confidence = float(probabilities[class_index])
                info_text = BRAIN_CLASS_INFO.get(result_text, "")

                st.markdown(
                    f"<div class='result-box {result_class}'>{result_text}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-title">Kısa Bilgi</div>
                        <p class="info-text">{info_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                confidence_percent = int(round(confidence * 100))
                st.write(f"**Güven Skoru:** %{confidence_percent}")
                st.progress(confidence_percent)
            else:
                st.info("Analiz için geçerli bir görüntü yükleyin.")

        if uploaded_file is not None:
            if st.button("Analiz Sonucunu Rapora Kaydet", key="brain-report"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.session_state["rapor_verileri"].append(
                    {
                        "baslik": "Beyin MR Analizi",
                        "Tarih": timestamp,
                        "Sonuç": result_text if image is not None else "Görüntü okunamadı",
                        "Güven Skoru": f"%{confidence_percent}" if image is not None else "-",
                    }
                )
                st.success("Rapor kaydedildi.")
    footer_ekle()


def render_liver_3d_section() -> None:
    render_section_header(
        "Karaciğer 3D Analizi",
        "https://assets3.lottiefiles.com/packages/lf20_j1adxtyb.json",
        height=160,
    )
    st.markdown(
        """
        <p class="subtitle">NifTI formatındaki 3D tomografi verilerini inceleyin ve PyTorch modeli ile analiz edin.</p>
        """,
        unsafe_allow_html=True,
    )

    if nib is None:
        st.error("nibabel yüklü olmadığı için NifTI dosyaları açılamıyor.")
        return
    if torch is None or AttentionUNet3D is None:
        st.error("PyTorch modeli yüklenemedi. Lütfen torch kurulumunu kontrol edin.")
        return

    uploaded_file = st.file_uploader(
        "NifTI dosyası yükleyin (.nii veya .nii.gz)",
        type=["nii", "nii.gz"],
        help="Karaciğer tomografisi için NifTI formatı desteklenir.",
        key="liver-3d-upload",
    )

    if uploaded_file is not None:
        try:
            volume_raw = load_nifti_volume(uploaded_file)
            volume_norm = normalize_liver_volume(volume_raw)
            st.session_state["liver_3d_volume"] = volume_norm
            st.session_state["liver_3d_filename"] = uploaded_file.name
            st.session_state["liver_3d_prediction"] = None
            st.session_state["liver_3d_has_tumor"] = None
        except Exception as exc:
            st.error(f"NifTI dosyası okunamadı: {exc}")

    volume = st.session_state.get("liver_3d_volume")
    if volume is None:
        st.info("Analiz için önce bir NifTI dosyası yükleyin.")
        footer_ekle()
        return

    opacity_percent = st.slider("Maske Şeffaflığı (Opacity)", 0, 100, 50)
    overlay_opacity = opacity_percent / 100.0

    def build_overlay_image(base_slice: np.ndarray, mask_slice: np.ndarray | None) -> np.ndarray:
        base_rot = np.rot90(base_slice, k=1)
        base_min = float(np.min(base_rot))
        base_max = float(np.max(base_rot))
        if base_max > base_min:
            base_norm = (base_rot - base_min) / (base_max - base_min)
        else:
            base_norm = np.zeros_like(base_rot, dtype=np.float32)
        base_rgb = np.stack([base_norm] * 3, axis=-1)

        if mask_slice is None:
            return (base_rgb * 255).astype(np.uint8)

        mask_rot = np.rot90(mask_slice, k=1)
        overlay_rgb = np.zeros_like(base_rgb, dtype=np.float32)
        overlay_alpha = np.zeros(base_rgb.shape[:2], dtype=np.float32)

        liver_mask = mask_rot == 1
        tumor_mask = mask_rot == 2

        overlay_rgb[liver_mask] = np.array([80, 200, 120], dtype=np.float32) / 255.0
        overlay_rgb[tumor_mask] = np.array([255, 59, 48], dtype=np.float32) / 255.0
        overlay_alpha[liver_mask | tumor_mask] = overlay_opacity

        overlay_alpha = overlay_alpha[..., None]
        blended = (base_rgb * (1.0 - overlay_alpha)) + (overlay_rgb * overlay_alpha)
        return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    def build_overlay_png_with_matplotlib(
        base_slice: np.ndarray,
        mask_slice: np.ndarray | None,
        alpha: float,
    ) -> io.BytesIO | np.ndarray:
        if plt is None or ListedColormap is None:
            return build_overlay_image(base_slice, mask_slice)

        base_rot = np.rot90(base_slice, k=1)
        mask_rot = np.rot90(mask_slice, k=1) if mask_slice is not None else None

        fig, ax = plt.subplots(figsize=(6, 6), dpi=140)
        ax.imshow(base_rot, cmap="gray")
        if mask_rot is not None:
            liver_mask = np.ma.masked_where(mask_rot != 1, mask_rot)
            tumor_mask = np.ma.masked_where(mask_rot != 2, mask_rot)
            ax.imshow(liver_mask, cmap=ListedColormap(["#50C878"]), alpha=alpha)
            ax.imshow(tumor_mask, cmap=ListedColormap(["#FF3B30"]), alpha=alpha)
        ax.axis("off")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buffer.seek(0)
        return buffer

    max_slice = int(volume.shape[0]) - 1
    selected_slice = st.slider(
        "Kesit Seçimi",
        min_value=0,
        max_value=max_slice,
        value=max_slice // 2,
        step=1,
    )

    pred_mask = st.session_state.get("liver_3d_prediction")
    mask_slice = pred_mask[selected_slice] if pred_mask is not None else None
    overlay_image = build_overlay_image(volume[selected_slice], mask_slice)
    st.image(
        overlay_image,
        caption="Secilen kesit (maske varsa boyandi)" if pred_mask is not None else "Secilen kesit",
        use_container_width=True,
    )

    if st.button("Yapay Zekayı Çalıştır", key="liver-3d-run"):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = load_liver_3d_model(str(LIVER_3D_MODEL_PATH), device_type)
        except Exception as exc:
            st.error(f"Model yüklenemedi: {exc}")
            footer_ekle()
            return

        if device_type == "cuda":
            torch.cuda.empty_cache()

        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        volume_tensor = volume_tensor.to(device_type)
        orig_d, orig_h, orig_w = volume_tensor.shape[-3:]

        patch_size = (128, 128, 128)
        stride = tuple(size // 2 for size in patch_size)

        pad_d = max(0, patch_size[0] - orig_d)
        pad_h = max(0, patch_size[1] - orig_h)
        pad_w = max(0, patch_size[2] - orig_w)

        if orig_d >= patch_size[0]:
            rem_d = (orig_d - patch_size[0]) % stride[0]
            pad_d = max(pad_d, (stride[0] - rem_d) % stride[0])
        if orig_h >= patch_size[1]:
            rem_h = (orig_h - patch_size[1]) % stride[1]
            pad_h = max(pad_h, (stride[1] - rem_h) % stride[1])
        if orig_w >= patch_size[2]:
            rem_w = (orig_w - patch_size[2]) % stride[2]
            pad_w = max(pad_w, (stride[2] - rem_w) % stride[2])

        if pad_d or pad_h or pad_w:
            volume_tensor = torch.nn.functional.pad(
                volume_tensor,
                (0, pad_w, 0, pad_h, 0, pad_d),
                mode="constant",
                value=0,
            )

        padded_d, padded_h, padded_w = volume_tensor.shape[-3:]
        output_logits = torch.zeros(
            (1, 3, padded_d, padded_h, padded_w),
            device=device_type,
            dtype=torch.float32,
        )
        count_map = torch.zeros(
            (1, 1, padded_d, padded_h, padded_w),
            device=device_type,
            dtype=torch.float32,
        )

        autocast_ctx = torch.amp.autocast("cuda") if device_type == "cuda" else nullcontext()
        with torch.no_grad():
            for d in range(0, padded_d - patch_size[0] + 1, stride[0]):
                for h in range(0, padded_h - patch_size[1] + 1, stride[1]):
                    for w in range(0, padded_w - patch_size[2] + 1, stride[2]):
                        patch = volume_tensor[
                            :, :, d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]
                        ]
                        with autocast_ctx:
                            logits = model(patch)
                        output_logits[
                            :, :, d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]
                        ] += logits
                        count_map[
                            :, :, d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]
                        ] += 1.0

        output_logits = output_logits / torch.clamp(count_map, min=1.0)
        preds = torch.argmax(output_logits, dim=1)

        pred_mask = preds.squeeze(0).detach().cpu().numpy()
        if pad_d or pad_h or pad_w:
            pred_mask = pred_mask[:orig_d, :orig_h, :orig_w]
        tumor_counts = np.sum(pred_mask == 2, axis=(1, 2))
        risk_slice = int(np.argmax(tumor_counts))
        tumor_in_volume = bool(np.any(pred_mask == 2))
        tumor_in_slice = bool(np.any(pred_mask[risk_slice] == 2))
        has_tumor = tumor_in_volume or tumor_in_slice

        st.session_state["liver_3d_prediction"] = pred_mask
        st.session_state["liver_3d_has_tumor"] = has_tumor
        st.session_state["liver_3d_risk_slice"] = risk_slice

        if device_type == "cuda":
            torch.cuda.empty_cache()

    has_tumor = st.session_state.get("liver_3d_has_tumor")
    risk_slice = st.session_state.get("liver_3d_risk_slice")
    if risk_slice is not None:
        st.markdown(
            f"**🤖 Yapay Zeka En Riskli Kesiti Tespit Etti: {risk_slice}. Dilim uzerinde tumor dokusu boyanmistir.**"
        )
    if has_tumor is not None:
        if has_tumor:
            st.error("⚠️ KARACİĞERDE TÜMÖR DOKUSU SAPTANDI")
            st.info("Dice Skoru: %41.71")
        else:
            st.success("Karaciğerde tümör dokusu saptanmadı.")

        if st.button("Analiz Sonucunu Rapora Kaydet", key="liver-3d-report"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state["rapor_verileri"].append(
                {
                    "baslik": "Karaciğer 3D Analizi",
                    "Tarih": timestamp,
                    "Dosya": st.session_state.get("liver_3d_filename", "-"),
                    "Sonuç": "Tümör dokusu saptandı" if has_tumor else "Temiz",
                    "Dice Skoru": "%41.71" if has_tumor else "-",
                }
            )
            st.success("Rapor kaydedildi.")

        if pred_mask is not None and risk_slice is not None:
            risk_base = volume[risk_slice]
            risk_mask = pred_mask[risk_slice]
            overlay_output = build_overlay_png_with_matplotlib(risk_base, risk_mask, overlay_opacity)
            st.image(
                overlay_output,
                caption=f"Boyalı en riskli kesit (#{risk_slice})",
                use_container_width=True,
            )

    footer_ekle()


def render_body_section() -> None:
    render_section_header(
        "Vücut Analizi (BMI/BMR)",
        "https://assets6.lottiefiles.com/packages/lf20_puciaact.json",
        height=150,
    )
    st.markdown(
        """
        <p class="subtitle">Boy, kilo ve yaşam tarzı bilgilerine göre sağlık hesaplamaları.</p>
        """,
        unsafe_allow_html=True,
    )
    input_col, output_col = st.columns([1, 1], gap="large")

    with input_col:
        st.subheader("Giriş Bilgileri")
        card_left, card_right = st.columns(2)
        with card_left:
            with st.container():
                st.markdown("**Temel Bilgiler**")
                weight_input = st.number_input(
                    "Kilo (kg)",
                    min_value=20.0,
                    max_value=300.0,
                    value=70.0,
                    step=0.5,
                )
                height_input = st.number_input(
                    "Boy (cm)",
                    min_value=100.0,
                    max_value=250.0,
                    value=170.0,
                    step=0.5,
                )
                age_input = st.number_input(
                    "Yaş",
                    min_value=10,
                    max_value=120,
                    value=30,
                    step=1,
                )
                gender_input = st.selectbox(
                    "Cinsiyet",
                    ["Erkek", "Kadın"],
                    index=0,
                )
                waist_cm = st.number_input(
                    "Bel çevresi (cm)",
                    min_value=40.0,
                    max_value=200.0,
                    value=80.0,
                    step=0.5,
                )
                hip_cm = st.number_input(
                    "Kalça çevresi (cm)",
                    min_value=50.0,
                    max_value=220.0,
                    value=100.0,
                    step=0.5,
                )
        with card_right:
            with st.container():
                st.markdown("**Yaşam Tarzı**")
                activity_level = st.selectbox(
                    "Aktivite Seviyesi",
                    ["Hareketsiz", "Hafif", "Orta", "Çok Hareketli"],
                    index=1,
                )
                water_intake = st.number_input(
                    "Günlük içilen su (L)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                )
                sleep_hours = st.slider("Günde kaç saat uyuyorsun?", 3.0, 10.0, 7.0, 0.5)
                stress_level = st.slider("Stres seviyen (1-10)", 1, 10, 5)
                exercise_days = st.slider("Haftada kaç gün egzersiz yapıyorsun?", 0, 7, 3)
                smoker = st.checkbox("Sigara kullanıyorum")
        with st.container():
            st.markdown("**Hedef ve Projeksiyon**")
            calorie_delta = st.slider("Günlük kalori farkı (kcal)", -700, 700, 0, 50)
            macro_goal = st.selectbox("Hedef", ["Zayıflama", "Kas Yapma", "Form Koruma"], index=2)
            macro_goal_dynamic = st.selectbox(
                "Hedefiniz nedir?",
                ["Kilo Ver", "Formu Koru", "Kas Yap"],
                index=1,
            )

    with output_col:
        st.subheader("Sonuçlar")

        bmi_value, bmi_category, bmi_note = calculate_bmi(weight_input, height_input)
        st.session_state["bmi_category"] = bmi_category
        st.session_state["bmi_value"] = bmi_value

        daily_water_need = calculate_daily_water_need(weight_input)

        whr_value = waist_cm / hip_cm if hip_cm > 0 else 0.0
        whr_risk = whr_value > (0.90 if gender_input == "Erkek" else 0.85)

        bmr_value = None
        total_calories = None
        try:
            bmr_value = calculate_bmr(weight_input, height_input, int(age_input), gender_input)
            activity_factors = {
                "Hareketsiz": 1.2,
                "Hafif": 1.375,
                "Orta": 1.55,
                "Çok Hareketli": 1.725,
            }
            total_calories = bmr_value * activity_factors.get(activity_level, 1.2)
        except Exception as exc:
            st.warning(f"BMR hesaplanamadı: {exc}")

        tab_basic, tab_comp, tab_nutrition = st.tabs(
            ["Temel Analiz", "Vücut Kompozisyonu", "Beslenme & Takip"]
        )

        with tab_basic:
            st.metric("BMI", f"{bmi_value:.1f}")
            st.caption(f"Kategori: {bmi_category} | {bmi_note}")
            if bmr_value is not None and total_calories is not None:
                bmr_col, calorie_col = st.columns(2)
                with bmr_col:
                    st.metric("BMR (kcal/gün)", f"{bmr_value:.0f}")
                with calorie_col:
                    st.metric("Günlük Toplam Kalori", f"{total_calories:.0f}")
                st.caption(
                    "BMR dinlenme halinde harcanan enerjiyi, toplam kalori ise aktiviteye göre günlük ihtiyacı ifade eder."
                )

            st.subheader("Gelecek Projeksiyonu")
            daily_delta = float(calorie_delta)
            six_month_days = 182
            one_year_days = 365
            delta_6m = (daily_delta * six_month_days) / 7700.0
            delta_1y = (daily_delta * one_year_days) / 7700.0
            projection_chart = {
                "Ay": [0, 6, 12],
                "Tahmini Kilo (kg)": [
                    float(weight_input),
                    float(weight_input) + delta_6m,
                    float(weight_input) + delta_1y,
                ],
            }
            st.line_chart(projection_chart, x="Ay", y="Tahmini Kilo (kg)")
            st.caption("7700 kcal yaklaşık 1 kg vücut ağırlığına eşdeğer kabul edilmiştir.")

            with st.expander("Detaylı Risk Analizi", expanded=False):
                risk_score = 0.0
                risk_score += max(0.0, min(40.0, (float(age_input) - 20.0) * 0.8))
                risk_score += max(0.0, min(30.0, (bmi_value - 22.0) * 2.0))
                if gender_input == "Erkek":
                    risk_score += 8.0
                if smoker:
                    risk_score += 15.0
                risk_score = float(np.clip(risk_score, 0.0, 100.0))
                st.progress(int(round(risk_score)))
                st.caption(f"Tahmini risk skoru: %{risk_score:.0f}")

        with tab_comp:
            st.metric("Bel/Kalça Oranı (WHR)", f"{whr_value:.2f}")
            if whr_risk:
                st.warning("Yüksek Risk (Elma Tipi)")
            else:
                st.info("Düşük Risk (Armut Tipi)")

            try:
                bio_age = calculate_bio_age(
                    int(age_input),
                    bmi_value,
                    float(sleep_hours),
                    int(stress_level),
                    int(exercise_days),
                )
                st.metric("Biyolojik Yaş", f"{bio_age:.1f}")
                st.caption(f"Vücudunuz yaklaşık {bio_age:.1f} yaşında hissediyor.")
            except Exception as exc:
                st.warning(f"Biyolojik yaş hesaplanamadı: {exc}")

            try:
                ideal_low, ideal_high = calculate_ideal_weight_range_miller(height_input, gender_input)
                if ideal_low <= weight_input <= ideal_high:
                    st.success(f"İdeal Kilo Aralığınız: {ideal_low:.1f} - {ideal_high:.1f} kg")
                else:
                    st.warning(f"İdeal Kilo Aralığınız: {ideal_low:.1f} - {ideal_high:.1f} kg")
            except Exception as exc:
                st.warning(f"İdeal kilo aralığı hesaplanamadı: {exc}")

        with tab_nutrition:
            st.metric("Günlük Su İhtiyacı", f"{daily_water_need:.2f} L")
            if total_calories is not None:
                st.subheader("Dinamik Makro Besin Dağılımı")
                dynamic_macro_map = {
                    "Kilo Ver": {"Protein": 40, "Karbonhidrat": 30, "Yağ": 30},
                    "Formu Koru": {"Protein": 30, "Karbonhidrat": 40, "Yağ": 30},
                    "Kas Yap": {"Protein": 30, "Karbonhidrat": 50, "Yağ": 20},
                }
                dynamic_macro = dynamic_macro_map.get(macro_goal_dynamic, dynamic_macro_map["Formu Koru"])
                fig_dynamic = go.Figure(
                    data=[
                        go.Bar(
                            x=list(dynamic_macro.keys()),
                            y=list(dynamic_macro.values()),
                            marker_color=["#0a6fb6", "#72b1e4", "#cfe6f7"],
                        )
                    ]
                )
                fig_dynamic.update_layout(
                    yaxis=dict(title="Oran (%)", range=[0, 60]),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_dynamic, use_container_width=True)
                dyn_protein_g = total_calories * (dynamic_macro["Protein"] / 100) / 4
                dyn_carb_g = total_calories * (dynamic_macro["Karbonhidrat"] / 100) / 4
                dyn_fat_g = total_calories * (dynamic_macro["Yağ"] / 100) / 9
                st.caption(
                    f"Hedefe gore dagilim: Protein {dyn_protein_g:.0f} g, Karbonhidrat {dyn_carb_g:.0f} g, Yag {dyn_fat_g:.0f} g."
                )
                macro_split = get_macro_distribution(macro_goal)
                labels = list(macro_split.keys())
                values = list(macro_split.values())
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45)])
                fig.update_traces(textinfo="label+percent")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                st.subheader("Makro Besin Dağılımı")
                st.plotly_chart(fig, use_container_width=True)
                protein_g = total_calories * (macro_split["Protein"] / 100) / 4
                carb_g = total_calories * (macro_split["Karbonhidrat"] / 100) / 4
                fat_g = total_calories * (macro_split["Yağ"] / 100) / 9
                st.caption(
                    f"Yaklaşık dağılım: Protein {protein_g:.0f} g, Karbonhidrat {carb_g:.0f} g, Yağ {fat_g:.0f} g."
                )

            recommendations = get_dynamic_recommendations(bmi_value, daily_water_need, water_intake)
            if recommendations:
                st.markdown("### Dinamik Tavsiyeler")
                for level, message in recommendations:
                    if level == "warning":
                        st.warning(message)
                    else:
                        st.success(message)

            st.subheader("Hidrasyon Takipcisi")
            if "water_ml" not in st.session_state:
                st.session_state["water_ml"] = 0
            if st.button("1 Bardak Ekle (250ml)", key="water-add"):
                st.session_state["water_ml"] = min(2500, st.session_state["water_ml"] + 250)
            water_progress = st.session_state["water_ml"] / 2500.0
            st.progress(int(round(water_progress * 100)))
            st.caption(f"Bugun: {st.session_state['water_ml']} ml / 2500 ml")

            with st.expander("Aktivite Onerileri", expanded=False):
                if total_calories is None:
                    st.info("Egzersiz karsiligi icin once BMR hesaplanmalidir.")
                else:
                    walk_hours = total_calories / 250.0
                    swim_hours = total_calories / 400.0
                    st.markdown(
                        f"Bu kaloriyi yakmak icin yaklasik {walk_hours:.1f} saat yuruyus veya {swim_hours:.1f} saat yuzme gerekir."
                    )

        if st.button("Verileri Rapora Kaydet", key="body-report"):
            st.session_state["rapor_verileri"].append(
                {
                    "baslik": "Vücut Analizi",
                    "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Boy (cm)": f"{height_input:.1f}",
                    "Kilo (kg)": f"{weight_input:.1f}",
                    "BMI": f"{bmi_value:.1f}",
                    "BMR": f"{bmr_value:.0f}" if "bmr_value" in locals() else "-",
                }
            )
            st.success("Rapor kaydedildi.")
    footer_ekle()


def render_guide_section() -> None:
    st.markdown(
        """
        <div class="app-shell">
            <h1 class="main-title">Sağlıklı Yaşam Rehberi</h1>
            <p class="subtitle">BMI sonucunuza göre kişisel öneriler ve günlük rutin ipuçları.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bmi_category = st.session_state.get("bmi_category")
    if not bmi_category:
        st.info("Rehber için önce 'Vücut Analizi (BMI/BMR)' sekmesinden bilgilerinizi girin.")
    else:
        st.markdown(f"**Mevcut BMI Kategoriniz:** {bmi_category}")

        with st.expander("Beslenme Önerileri"):
            if bmi_category in {"Fazla Kilolu", "Obez"}:
                row_1, row_2, row_3 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("💧")
                with row_2:
                    st.markdown("Karbonhidrat alımını azaltın, lifli gıdaları artırın.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🍎")
                with row_2:
                    st.markdown("Şekerli içeceklerden kaçının, su tüketimini yükseltin.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("⚖️")
                with row_2:
                    st.markdown("Porsiyon kontrolüne odaklanın.")
            elif bmi_category == "Zayıf":
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🥩")
                with row_2:
                    st.markdown("Protein ve sağlıklı yağ alımını artırın.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("⚡")
                with row_2:
                    st.markdown("Günlük öğün sayısını artırarak kalori açığı kapatın.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🥛")
                with row_2:
                    st.markdown("Kuruyemiş ve süt ürünlerinden destek alın.")
            else:
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🥗")
                with row_2:
                    st.markdown("Dengeli karbonhidrat-protein dağılımını koruyun.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🍊")
                with row_2:
                    st.markdown("Haftalık sebze ve meyve çeşitliliğini artırın.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🚫")
                with row_2:
                    st.markdown("İşlenmiş gıdaları sınırlayın.")

        with st.expander("Basit Egzersiz Önerileri"):
            if bmi_category in {"Fazla Kilolu", "Obez"}:
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🚶")
                with row_2:
                    st.markdown("Günde 30 dakika tempolu yürüyüş.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🏋️")
                with row_2:
                    st.markdown("Hafif direnç egzersizleri (haftada 2-3 gün).")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🤸")
                with row_2:
                    st.markdown("Kısa esneme rutinleri ile hareketliliği artırın.")
            elif bmi_category == "Zayıf":
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("💪")
                with row_2:
                    st.markdown("Kas kütlesi için vücut ağırlığıyla güç egzersizleri.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("⏱️")
                with row_2:
                    st.markdown("Kısa ama düzenli direnç antrenmanları.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("📅")
                with row_2:
                    st.markdown("Haftalık 3 gün hedefleyin.")
            else:
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🏃")
                with row_2:
                    st.markdown("Haftada 150 dakika orta tempo kardiyo.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("🧘")
                with row_2:
                    st.markdown("Denge ve core egzersizleri ekleyin.")
                row_1, row_2 = st.columns([0.12, 0.88])
                with row_1:
                    st.markdown("👟")
                with row_2:
                    st.markdown("Günlük adım sayısını artırın.")

    st.markdown("### İyileşme Kronometresi")
    quit_hours = st.number_input("Sigara bırakalı kaç saat oldu?", min_value=0, max_value=20000, value=0, step=1)
    milestones = [
        (0.33, "20 dakika", "Kan basıncı ve nabız normale dönmeye başlar."),
        (24, "24 saat", "Karbonmonoksit seviyesi düşer."),
        (48, "48 saat", "Koku ve tat duyuları iyileşir."),
        (336, "2 hafta", "Dolaşım ve akciğer fonksiyonları iyileşir."),
        (2160, "3 ay", "Öksürük ve nefes darlığı azalır."),
    ]
    for hours, label, note in milestones:
        status = "Tamamlandı" if quit_hours >= hours else "Bekleniyor"
        st.markdown(f"- **{label}:** {note} _(Durum: {status})_")
    footer_ekle()


def render_info_section() -> None:
    st.markdown(
        """
        <div class="app-shell">
            <h1 class="main-title">Bilgi Bankası</h1>
            <p class="subtitle">Projede kullanılan teknik ve tıbbi terimlerin profesyonel sözlüğü.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Bu bölüm, jürinin ve proje inceleyenlerin teknik akışı doğru anlaması için hazırlanmış kısa bir sözlük alanıdır."
    )

    glossary_sections = [
        (
            "🏥 1. Klinik ve Tıbbi Terminoloji",
            [
                (
                    "NifTI (.nii / .nii.gz)",
                    "Neuroimaging Informatics Technology Initiative açılımına sahip, BT ve MR gibi medikal görüntülerin hastanın anatomisini 3 boyutlu bir hacim olarak sakladığı gelişmiş bir tıbbi dosya formatıdır.",
                ),
                (
                    "Bilgisayarlı Tomografi (CT / BT)",
                    "X-ışınları kullanılarak vücudun incelenen bölgesinin kesitler halinde 3 boyutlu görüntüsünü oluşturan radyolojik tanı yöntemidir.",
                ),
                (
                    "Karaciğer Parankimi",
                    "Karaciğerin görev yapan, fonksiyonel ana dokusudur. Yapay zekamız tümörü ararken önce bu sağlıklı dokunun sınırlarını tespit eder.",
                ),
                (
                    "Lezyon / Tümör Dokusu",
                    "Organ içerisinde meydana gelen, normal hücre yapısından farklılaşmış, kontrolsüz büyüyen hastalık veya kitle dokusudur.",
                ),
                (
                    "Aksiyel / Sagittal Kesit",
                    "Tomografide vücuda yukarıdan aşağıya yatay bakılan dilimlere aksiyel, yandan omurga boyunca dik bakılan dilimlere sagittal kesit denir.",
                ),
            ],
        ),
        (
            "🦾 2. Yapay Zeka ve Derin Öğrenme Terminolojisi",
            [
                (
                    "Attention U-Net 3D",
                    "Görüntü segmentasyonunda kullanılan bir derin öğrenme mimarisidir. Attention mekanizması sayesinde model arka plandaki gereksiz dokuları görmezden gelerek sadece karaciğer ve tümör piksellerine odaklanır.",
                ),
                (
                    "Segmentasyon (Görüntü Bölütleme)",
                    "Yapay zekanın görüntüdeki nesneleri piksellerine kadar inceleyip sınırlarını çizerek renkli katmanlarla birbirinden ayırması işlemidir. Örneğin karaciğeri yeşile, tümörü kırmızıya boyamak bu kapsamdadır.",
                ),
                (
                    "Inference (Tahmin / Çıkarım)",
                    "Eğitilmiş bir yapay zeka modelinin daha önce hiç görmediği yeni bir hasta tomografisini analiz ederek sonuç üretmesi aşamasıdır.",
                ),
                (
                    "VRAM (Video RAM)",
                    "Ekran kartı üzerinde bulunan ultra hızlı geçici hafızadır. 3D tomografi matrisleri çok büyük olduğundan tahmin esnasında bu hafıza sınırları zorlanır.",
                ),
                (
                    "Sliding Window Inference",
                    "Devasa 3D tomografi dosyalarının tek seferde VRAM'i zorlamasını önlemek için görüntüyü küçük parçalara bölerek yapay zekaya işletme ve sonrasında bu parçaları birleştirme tekniğidir.",
                ),
            ],
        ),
        (
            "📊 3. Başarı ve Performans Metrikleri",
            [
                (
                    "Dice Skoru (Sorenson-Dice)",
                    "Yapay zekanın boyadığı alan ile doktorun boyadığı alanın ne kadar üst üste bindiğini ölçen, %0 ile %100 arasında değer alan temel medikal başarı metriğidir.",
                ),
                (
                    "Precision (Kesinlik)",
                    "Yapay zekanın tümör dediği piksellerin yüzde kaçının gerçekten tümör olduğunu gösteren metriktir; yanlış alarm oranını ölçer.",
                ),
                (
                    "Recall (Duyarlılık / Hassasiyet)",
                    "Hastadaki gerçek tümör piksellerinin yüzde kaçının yapay zeka tarafından başarıyla yakalanabildiğini gösteren hayati tıbbi metriktir.",
                ),
            ],
        ),
    ]

    for section_title, items in glossary_sections:
        st.markdown(f"### {section_title}")
        for term, description in items:
            with st.expander(term, expanded=False):
                st.markdown(description)
                st.info(f"**{term}:** {description}")

    footer_ekle()


def render_support_section() -> None:
    st.markdown(
        """
        <div class="app-shell">
            <h1 class="main-title">Destek ve İletişim</h1>
            <p class="subtitle">Yakın sağlık kuruluşları ve randevu yönlendirmeleri.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("En Yakın Sağlık Kuruluşu")
    st.markdown(
        """
        - Karabük Eğitim ve Araştırma Hastanesi
        - Safranbolu Devlet Hastanesi
        - Yenice Devlet Hastanesi
        """
    )

    if st.button("MHRS Üzerinden Randevu Al"):
        st.markdown("[MHRS Randevu Sistemi](https://mhrs.gov.tr/)")

    st.error("Bu bir yapay zeka asistanıdır, acil durumlar için lütfen 112'yi arayın.")
    footer_ekle()


def render_lab_section() -> None:
    st.markdown(
        """
        <div class="app-shell">
            <h1 class="main-title">Laboratuvar Analiz Rehberi</h1>
            <p class="subtitle">Kan tahlili sonuçlarını referans aralıklarına göre değerlendirin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_titles = list(LAB_REFERENCE_RANGES.keys())
    tabs = st.tabs(tab_titles)
    abnormal_records: list[tuple[str, float, str]] = []

    for tab, category in zip(tabs, tab_titles, strict=False):
        with tab:
            for name, meta in LAB_REFERENCE_RANGES[category].items():
                value = st.number_input(
                    f"{name} ({meta['unit']})",
                    min_value=0.0,
                    value=float(meta["min"]),
                    step=0.1,
                    key=f"lab_{name}",
                )
                st.caption(meta["help"])

                status = "Normal ✅"
                if value < meta["min"]:
                    status = "Düşük ⚠️"
                    st.warning(status)
                elif value > meta["max"]:
                    status = "Yüksek 🚨"
                    st.error(status)
                else:
                    st.success(status)

                if status != "Normal ✅":
                    abnormal_records.append((name, value, status))

    if st.button("Laboratuvar Sonuclarini Rapora Ekle", key="lab-report"):
        added_count = 0
        for name, value, status in abnormal_records:
            record = f"{name} - {value:.2f} - {status}"
            if record not in st.session_state["lab_rapor_kayitlari"]:
                st.session_state["lab_rapor_kayitlari"].add(record)
                st.session_state["rapor_verileri"].append(
                    {
                        "baslik": "Laboratuvar Analiz Rehberi",
                        "Detay": record,
                    }
                )
                added_count += 1
        if added_count:
            st.success(f"Rapor eklendi: {added_count} bulgu")
        else:
            st.info("Rapora eklenecek yeni bulgu bulunamadi.")

    st.info("Bu sonuçlar sadece bilgilendirme amaçlıdır.")
    footer_ekle()


def render_report_section() -> None:
    st.markdown(
        """
        <div class="app-shell">
            <h1 class="main-title">Sağlık Raporu</h1>
            <p class="subtitle">Kaydedilen analiz sonuçlarının özeti.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Raporu Temizle 🗑️", type="primary"):
        st.session_state["rapor_verileri"] = []
        st.session_state["lab_rapor_kayitlari"] = set()
        st.experimental_rerun()

    entries = st.session_state.get("rapor_verileri", [])
    if not entries:
        st.warning("Henüz kaydedilmiş veri yok.")
        return

    def is_risk_value(label: str, value: str) -> bool:
        risk_terms = [
            "ZATÜRRE",
            "PNEUMONIA",
            "Glioma",
            "Meningioma",
            "Pituitary",
            "Tümör",
            "Düşük",
            "Yüksek",
            "Risk",
        ]
        if label == "BMI":
            try:
                bmi_val = float(str(value))
                return bmi_val < 18.5 or bmi_val >= 25.0
            except ValueError:
                return False
        text = str(value)
        return any(term in text for term in risk_terms)

    for item in entries:
        with st.container():
            st.markdown("---")
            st.markdown(f"**{item.get('baslik', 'Kayıt')}**")
            if "Tarih" in item:
                st.caption(f"Tarih: {item['Tarih']}")

            metric_items = [(k, v) for k, v in item.items() if k not in {"baslik", "Tarih"}]
            if not metric_items:
                continue

            chunk_size = 3
            for idx in range(0, len(metric_items), chunk_size):
                chunk = metric_items[idx:idx + chunk_size]
                cols = st.columns(len(chunk))
                for col, (key, value) in zip(cols, chunk, strict=False):
                    risk = is_risk_value(key, value)
                    delta_label = "Risk" if risk else "Normal"
                    delta_color = "inverse" if risk else "normal"
                    with col:
                        st.metric(
                            label=key,
                            value=str(value),
                            delta=delta_label,
                            delta_color=delta_color,
                        )

    pdf_bytes = create_pdf_report(entries)
    st.download_button(
        "PDF Olarak İndir",
        data=pdf_bytes,
        file_name="klinik_analiz_raporu.pdf",
        mime="application/pdf",
    )
    footer_ekle()


def render_about_section() -> None:
    st.markdown(
        """
        <div class="app-shell">
            <h1 class="main-title">Proje Künyesi ve Hakkında</h1>
            <p class="subtitle">Projenin amacı, kapsamı ve iletişim bilgileri.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    content_col, logo_col = st.columns([3, 1])
    with content_col:
        st.markdown("### Kişisel ve Akademik Bilgiler")
        st.markdown("**Ad-Soyad:** Eren Özkan")
        st.markdown("**Bölüm:** Karabük Üniversitesi - Yapay Zeka Operatörlüğü")
        st.markdown(
            "**Proje Amacı:** Bu çalışma, 2026 Haziran dönemi mezuniyet projesi olarak; "
            "yapay zeka tekniklerinin medikal teşhis ve kişisel sağlık yönetimindeki potansiyelini "
            "sergilemek amacıyla geliştirilmiştir."
        )

        st.markdown("### İletişim Paneli")
        st.markdown("**Sistemi geliştirmem için tavsiye ve önerilerinizi paylaşabilirsiniz.**")
        st.info("📧 İletişim: ernozkn78@gmail.com")
        st.link_button("📧 Mail Gönder", "mailto:ernozkn78@gmail.com")

    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), use_container_width=True)
        else:
            st.warning("Logo bulunamadı: Proje_logo.png")

    footer_ekle()


if menu_selection == "Ana Sayfa 🏠":
    render_home_section()
elif menu_selection == "Akciğer Analizi 🫁":
    render_lung_section()
elif menu_selection == "Beyin MR Analizi 🧠":
    render_brain_section()
elif menu_selection == "Karaciğer 3D Analizi 🚀":
    render_liver_3d_section()
elif menu_selection == "Vücut Analizi (BMI/BMR) 🏋️‍♂️":
    render_body_section()
elif menu_selection == "Laboratuvar Analiz Rehberi 🧪":
    render_lab_section()
elif menu_selection == "Sağlıklı Yaşam Rehberi 🍎":
    render_guide_section()
elif menu_selection == "Sağlık Raporu 📄":
    render_report_section()
elif menu_selection == "Bilgi Bankası 📗":
    render_info_section()
elif menu_selection == "Destek ve İletişim 📞":
    render_support_section()
elif menu_selection == "Hakkında ℹ️":
    render_about_section()
else:
    render_support_section()

st.markdown(
    """
    <div class="legal-note">
        Bu bir yapay zeka destekli analiz aracıdır. Kesin teşhis için lütfen bir radyolog görüşü alınız.
    </div>
    """,
    unsafe_allow_html=True,
)
