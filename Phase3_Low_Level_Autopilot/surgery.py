import os
import torch
from stable_baselines3 import SAC
from models.sac_actor import SafeActor

def perform_network_surgery():
    print("="*60)
    print("🧠 YAPAY ZEKA CERRAHİSİ BAŞLIYOR (11 Sensör -> 14 Sensör)")
    print("="*60)

    # 1. ESKİ MODELİ YÜKLE (Faz 2'den kalan .zip dosyanın yolu)
    # NOT: Kendi Faz 2 modelinin tam adını ve yolunu buraya yaz!
    old_model_path = "./fighter_checkpoints/phase2_vanilla/sac_f16_fighter_phase2_vanilla_final.zip"
    
    if not os.path.exists(old_model_path):
        print(f"[HATA] Eski model bulunamadı: {old_model_path}")
        print("Lütfen dosya yolunu kontrol et.")
        return

    print(f"1. Eski beyin (.zip) masaya yatırıldı: {old_model_path}")
    old_model = SAC.load(old_model_path, device="cpu")
    old_actor_dict = old_model.actor.state_dict()

    # 2. YENİ MODELİ OLUŞTUR (14 Sensörlü ve CBF Kalkanlı yeni yapımız)
    print("2. Yeni beyin iskeleti (14 Boyutlu SafeActor) hazırlanıyor...")
    new_actor = SafeActor(state_dim=14, action_dim=3)
    new_actor_dict = new_actor.state_dict()

    # 3. SİNİR KATMANLARINI EŞLEŞTİRME (Mapping)
    # SB3'ün iç mimarisindeki katman isimleri ile bizim kendi yazdığımız katman isimleri
    layer_mapping = {
        'latent_pi.0.weight': 'net.0.weight',  # İlk Katman Ağırlıkları (11 -> 14 olacak)
        'latent_pi.0.bias':   'net.0.bias',    # İlk Katman Biasları
        'latent_pi.2.weight': 'net.2.weight',  # İkinci Katman (Gizli)
        'latent_pi.2.bias':   'net.2.bias',
        'mu.weight':          'net.4.weight',  # Çıkış Katmanı (Aksiyon)
        'mu.bias':            'net.4.bias'
    }

    # 4. AMELİYAT BAŞLIYOR (Transfer Learning)
    for old_key, new_key in layer_mapping.items():
        if old_key not in old_actor_dict:
            continue
            
        old_tensor = old_actor_dict[old_key]
        new_tensor = new_actor_dict[new_key]
        
        # A) DOĞRUDAN KOPYALAMA (Boyutları aynı olan iç katmanlar)
        if old_tensor.shape == new_tensor.shape:
            new_actor_dict[new_key] = old_tensor.clone()
            print(f"  [BAŞARILI] {old_key} -> {new_key} (Kusursuz aktarıldı)")
            
        # B) CERRAHİ GENİŞLETME (11'den 14'e çıkan ilk katman ağırlıkları)
        elif "weight" in old_key and old_tensor.shape[1] == 11 and new_tensor.shape[1] == 14:
            with torch.no_grad(): # Türev (Gradient) takibini kapatıyoruz ki manuel değer atayabilelim
                # 1. Eski 11 nöronluk uçuş tecrübesini (kas hafızasını) yeni ağın ilk 11 nöronuna kopyala
                new_actor_dict[new_key][:, :11] = old_tensor.clone()
                
                # 2. Yeni eklenen 3 sensöre (Es, Yakıt, Hedefler) SIFIR (0.0) ağırlık ver.
                # Neden Sıfır? Çünkü bu sensörleri ilk gördüğünde tepki vermesin, yavaş yavaş öğrensin.
                new_actor_dict[new_key][:, 11:] = 0.0
                
            print(f"  [AMELİYAT] {old_key} -> {new_key} (11 Nöron başarıyla 14'e genişletildi. Yeni nöronlar uyutuldu!)")
        
        else:
            print(f"  [UYARI] Beklenmeyen boyut uyumsuzluğu: {old_key} -> {new_key}")

    # 5. YENİ BEYNİ ENJEKTE ET VE KAYDET
    new_actor.load_state_dict(new_actor_dict)
    
    # Faz 3'ün kayıt klasörü yoksa oluştur
    save_dir = "./fighter_checkpoints/phase3_cbf"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "sac_actor_transferred_14_sensors.pth")
    torch.save(new_actor.state_dict(), save_path)
    
    print("="*60)
    print(f"🎉 OPERASYON TAMAMLANDI! Yeni Aktör hazır: {save_path}")
    print("Art train.py dosyasındaki LOAD_MODEL_PATH_ACTOR değişkenine bu yolu verebilirsin!")
    print("="*60)

if __name__ == "__main__":
    perform_network_surgery()