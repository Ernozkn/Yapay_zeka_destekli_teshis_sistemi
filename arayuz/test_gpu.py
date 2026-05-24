import torch

print("=========================================")
print(f"PyTorch Sürümü: {torch.__version__}")
print(f"CUDA (GPU) Aktif mi?: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Kullanılabilir GPU Sayısı: {torch.cuda.device_count()}")
    print(f"Aktif Ekran Kartı: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Sürümü: {torch.version.cuda}")
else:
    print("❌ DİKKAT: Ekran kartı aktif DEĞİL! CPU modundasın.")
print("=========================================")