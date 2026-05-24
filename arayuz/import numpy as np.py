import os # İşletim sistemiyle iletişim kurmanı sağlar
import re # Metinler içinde karmaşık aramalar ve düzenlemeler yapmanızı sağlar.
import torch # Yapay zeka modellerinin matematiksel hesaplamalarını yapar.
from diffusers import AutoPipelineForText2Image # Metinden görüntü oluşturan (ST)
from huggingface_hub.utils import HfHubHTTPError # Programın çökmesini engelleme

# model havuzu
CANDIDATE_MODELS = [
    ("prompthero/openjourney", "OpenJourney (Midjourney v4 benzeri stil)"),
    ("dreamlike-art/dreamlike-diffusion-1.0", "Dreamlike Diffusion (fantazi/dijital sanat)"),
    ("runwayml/stable-diffusion-v1-5", "Stable Diffusion 1.5 (bazı hesaplarda gated olabilir)"),
    ("stabilityai/sd-turbo", "SD-Turbo (hızlı, bazı hesaplarda gated olabilir)")
]

def safe_filename(s: str) -> str:
    s = s.strip().lower() # Metnin başındaki ve sonundaki gereksiz boşlukları siler ve küçük harfe çevirir.
    s = re.sub(r"\s+", "_", s) # Metin içindeki tüm boşlukları (veya yan yana gelen boşlukları) alt çizgiye çevirir.
    s = re.sub(r"[^a-z0-9_\-]+", "", s) # Bu bir "temizlik" aşamasıdır. Küçük harf, rakam, alt çizgi ve tire dışındakileri siler.
    return (s[:80] or "image") + ".png" # Dosya isminin çok uzun olup sistemi yormasını engeller ve sonuna .png ekler.

def ask(prompt, default=None): # Fonksiyon iki parametre alır: prompt, default
    if default is None: # Standart bir input çalıştırır