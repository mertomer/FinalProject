import requests
import datetime
import time

# --- AYARLAR ---

# 1. GÜVENLİK UYARISI: Lütfen Etherscan'den YENİ bir API anahtarı alın ve buraya yapıştırın.
#    Daha önce paylaştığınız anahtar artık güvenli değildir!
API_KEY = "RITCEX7USMJGW2G95SFBJR5RFG8MVM3K6Z"

# 2. Verisini çekmek istediğiniz tarihi ayarlayın
TARGET_YEAR = 2024
TARGET_MONTH = 10
TARGET_DAY = 1  # 1 Ekim

# --- Etherscan API V2 ---
BASE_URL = "https://api.etherscan.io/v2/api"

def get_block_number_by_time(timestamp, closest="after", api_key=API_KEY):
    """
    Belirtilen bir Unix zaman damgasına en yakın blok numarasını
    Etherscan API'sini kullanarak çeker.
    """
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": timestamp,
        "closest": closest,
        "chainid": "1",
        "apikey": api_key
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # HTTP hatası varsa fırlat
        data = response.json()
        
        if data["status"] == "1":
            return data["result"]
        else:
            print(f"API Hatası ({closest}): {data['message']} (Result: {data['result']})")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API isteği sırasında bir hata oluştu: {e}")
        return None

# --- Ana İşlem ---
if __name__ == "__main__":
    if API_KEY == "YENI_API_KEYINIZI_BURAYA_GIRIN":
        print("HATA: Lütfen `find_blocks.py` dosyasını açıp API_KEY değişkenini güncelleyin.")
    else:
        # 1. Başlangıç ve bitiş zaman damgalarını hesapla
        start_dt = datetime.datetime(TARGET_YEAR, TARGET_MONTH, TARGET_DAY, 0, 0, 0)
        start_timestamp = int(start_dt.timestamp())
        
        end_dt = datetime.datetime(TARGET_YEAR, TARGET_MONTH, TARGET_DAY, 23, 59, 59)
        end_timestamp = int(end_dt.timestamp())

        print(f"{start_dt.date()} tarihi için blok numaraları aranıyor...")
        print(f"Başlangıç Zaman Damgası: {start_timestamp}")
        print(f"Bitiş Zaman Damgası: {end_timestamp}")
        print("-" * 30)

        # 2. API'ye istekleri gönder
        
        # Günün İLK BLOĞU: 00:00:00'dan 'sonra' (after) gelen ilk blok
        start_block = get_block_number_by_time(start_timestamp, "after")
        
        # Etherscan'in saniyede 5 istek limitine takılmamak için 1 saniye bekle
        time.sleep(1) 
        
        # Günün SON BLOĞU: 23:59:59'dan 'önce' (before) gelen son blok
        end_block = get_block_number_by_time(end_timestamp, "before")

        print("-" * 30)
        
        if start_block and end_block:
            print("Blok numaraları başarıyla bulundu.")
            print("Lütfen aşağıdaki numaraları kopyalayıp `fetch_data.py` dosyasına yapıştırın:\n")
            print(f"[OK] BAŞLANGIÇ BLOĞU (START_BLOCK): {start_block}")
            print(f"[OK] BİTİŞ BLOĞU (END_BLOCK):    {end_block}")
            
            total_blocks = int(end_block) - int(start_block) + 1
            print(f"\nBu tarihte toplam {total_blocks} blok işlenecek.")
            
        else:
            print("Blok numaraları alınamadı. API anahtarınızı kontrol edin.")