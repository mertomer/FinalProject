# eth_analyzer/block_finder.py
import requests
import datetime
import time
import sys
import os
import json

# Eğer bu script doğrudan çalıştırılıyorsa, eth_analyzer paketini Python yoluna ekle
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

# config'den gerekli ayarları import et
from eth_analyzer.config import (
    ETHERSCAN_API_KEY, ETHERSCAN_BASE_URL,
    DEFAULT_TARGET_YEAR, DEFAULT_TARGET_MONTH, DEFAULT_TARGET_DAY
)

def get_block_number_by_time(timestamp, closest="after"):
    """
    Belirtilen bir Unix zaman damgasına en yakın blok numarasını çeker.
    API Anahtarı config dosyasından alınır.
    """
    if not ETHERSCAN_API_KEY or ETHERSCAN_API_KEY == "YENI_API_KEYINIZI_BURAYA_GIRIN":
         print("HATA: Etherscan API anahtarı config.py dosyasında ayarlanmamış.")
         return None

    params = {
        "module": "block", "action": "getblocknobytime",
        "timestamp": timestamp, "closest": closest,
        "chainid": "1", "apikey": ETHERSCAN_API_KEY
    }
    try:
        response = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=10) # Timeout eklendi
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "1":
            return data.get("result")
        else:
            print(f"Etherscan API Hatası ({closest}): {data.get('message', 'Bilinmeyen')} (Result: {data.get('result', '')})")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Etherscan API isteği sırasında hata: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Etherscan API'den geçersiz JSON yanıtı alındı.")
        return None

def find_block_range_for_date(year=DEFAULT_TARGET_YEAR, month=DEFAULT_TARGET_MONTH, day=DEFAULT_TARGET_DAY):
    """
    Belirtilen tarih için başlangıç ve bitiş blok numaralarını bulur.
    Başarılı olursa (start_block, end_block) tuple'ı, olmazsa (None, None) döndürür.
    """
    try:
        start_dt = datetime.datetime(year, month, day, 0, 0, 0)
        start_timestamp = int(start_dt.timestamp())
        end_dt = datetime.datetime(year, month, day, 23, 59, 59)
        end_timestamp = int(end_dt.timestamp())

        print(f"\n{start_dt.date()} tarihi için Etherscan'den blok aralığı aranıyor...")

        start_block_str = get_block_number_by_time(start_timestamp, "after")
        if start_block_str is None: return None, None # Hata durumunda çık

        print("İlk blok numarası alındı, 1 saniye bekleniyor...")
        time.sleep(1.1) # Etherscan rate limit için biraz daha fazla bekle

        end_block_str = get_block_number_by_time(end_timestamp, "before")
        if end_block_str is None: return None, None # Hata durumunda çık

        try:
            start_block = int(start_block_str)
            end_block = int(end_block_str)
            if start_block > end_block:
                 print(f"HATA: Başlangıç bloğu ({start_block}) bitiş bloğundan ({end_block}) büyük olamaz.")
                 return None, None

            print(f"✅ Blok aralığı bulundu: {start_block} - {end_block}")
            return start_block, end_block
        except (ValueError, TypeError):
            print(f"HATA: Etherscan'den dönen blok numaraları geçersiz: {start_block_str}, {end_block_str}")
            return None, None

    except ValueError:
        print(f"HATA: Geçersiz tarih: {year}-{month}-{day}")
        return None, None
    except Exception as e:
        print(f"Blok aralığı bulunurken beklenmedik hata: {e}")
        return None, None

# Bu dosyanın doğrudan çalıştırılması için test bloğu
if __name__ == "__main__":
    print("Bu script normalde import edilerek kullanılır.")
    print("Test amacıyla varsayılan tarih için blok aralığı bulunuyor...")
    start_b, end_b = find_block_range_for_date()
    if start_b and end_b:
        print(f"\nTest Başarılı. Bulunan aralık: {start_b} - {end_b}")
    else:
        print("\nTest Başarısız. Blok aralığı bulunamadı.")