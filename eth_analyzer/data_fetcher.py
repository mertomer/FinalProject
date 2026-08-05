# eth_analyzer/data_fetcher.py
import sqlite3
import requests
import time
import json
import sys
import os

# Eğer bu script doğrudan çalıştırılıyorsa, eth_analyzer paketini Python yoluna ekle
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

# config'den gerekli ayarları import et
from eth_analyzer.config import ALCHEMY_RPC_URL, REQUEST_HEADERS, DB_FILE_PATH
# block_finder'ı da import et (doğrudan çalıştırma için)
from eth_analyzer.block_finder import find_block_range_for_date

# --- VERİTABANI İŞLEMLERİ ---
def setup_database(db_path=DB_FILE_PATH):
    """Veritabanını ve tabloları belirtilen yolda oluşturur."""
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir): # db_dir boş değilse kontrol et
            os.makedirs(db_dir)
            print(f"'{db_dir}' dizini oluşturuldu.")

        conn = sqlite3.connect(db_path, timeout=20)
        cursor = conn.cursor()

        # Blocks Tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Blocks (
            blockNumber INTEGER PRIMARY KEY, timestamp INTEGER, hash TEXT NOT NULL,
            miner TEXT, gasUsed INTEGER, gasLimit INTEGER, transactionCount INTEGER
        )''')
        # Transactions Tablosu
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Transactions (
            txHash TEXT PRIMARY KEY, blockNumber INTEGER, fromAddress TEXT,
            toAddress TEXT, value_wei TEXT, gasPrice TEXT, gas TEXT
        )''')

        conn.commit()
        print(f"Veritabanı '{os.path.basename(db_path)}' başarıyla kuruldu/yüklendi.")
        return conn
    except sqlite3.Error as e:
        print(f"Veritabanı kurulum hatası ({db_path}): {e}")
        return None
    except OSError as e:
        print(f"Dizin oluşturma hatası ({db_dir}): {e}")
        return None

# --- API İSTEĞİ ---
def get_block_data_robust(block_number, max_retries=5, initial_wait_sec=2):
    """Bir blok verisini çeker (Robust)."""
    hex_block_number = hex(block_number)
    if not hex_block_number.startswith('0x'): hex_block_number = '0x' + hex_block_number[2:]
    payload = {"jsonrpc": "2.0", "method": "eth_getBlockByNumber", "params": [hex_block_number, True], "id": 1}
    wait_time = initial_wait_sec
    for attempt in range(max_retries):
        try:
            response = requests.post(ALCHEMY_RPC_URL, json=payload, headers=REQUEST_HEADERS, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data and "result" in data and data["result"]: return data["result"]
            else: print(f"Blok {block_number}, Deneme {attempt + 1}: RPC Hatası: {data.get('error', 'Bilinmeyen')}")
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429: print(f"Blok {block_number}: Rate Limit (429). {wait_time}s bekleniyor...")
            else: print(f"Blok {block_number}: HTTP Hatası: {http_err.response.status_code} - {http_err.response.text}")
        except requests.exceptions.RequestException as e: print(f"Blok {block_number}: Bağlantı Hatası: {e}")
        except json.JSONDecodeError: print(f"Blok {block_number}: Geçersiz JSON yanıtı alındı.")

        if attempt < max_retries - 1: time.sleep(wait_time); wait_time *= 2
    print(f"HATA: Blok {block_number} {max_retries} deneme sonunda alınamadı."); return None

# --- VERİ İŞLEME ---
def process_block(conn, block_data):
    """Gelen blok verisini veritabanına işler."""
    if not block_data or 'number' not in block_data:
        print("Geçersiz blok verisi alındı, işlenemiyor.")
        return False
    block_num_int = int(block_data['number'], 16) # Hata mesajları için

    cursor = conn.cursor()
    try:
        # Blok verisi None kontrolü eklendi
        block_tuple = (
            block_num_int,
            int(block_data.get('timestamp', '0x0'), 16),
            block_data.get('hash', ''),
            block_data.get('miner', ''),
            int(block_data.get('gasUsed', '0x0'), 16),
            int(block_data.get('gasLimit', '0x0'), 16),
            len(block_data.get('transactions', []))
        )
        cursor.execute('INSERT OR IGNORE INTO Blocks VALUES (?, ?, ?, ?, ?, ?, ?)', block_tuple)

        transactions_to_insert = []
        for tx in block_data.get('transactions', []):
            # Anahtar alanların varlığını kontrol et
            if not all(k in tx for k in ('hash', 'blockNumber', 'from', 'value', 'gasPrice', 'gas')):
                print(f"Blok {block_num_int} içinde eksik veri içeren işlem atlanıyor: {tx.get('hash', 'HASH YOK')}")
                continue
            to_address = tx.get('to') or "CONTRACT_CREATION"
            transactions_to_insert.append((
                tx['hash'], int(tx['blockNumber'], 16), tx['from'], to_address,
                tx['value'], tx['gasPrice'], tx['gas']
            ))

        if transactions_to_insert: # Sadece ekleyecek işlem varsa çalıştır
            cursor.executemany('INSERT OR IGNORE INTO Transactions VALUES (?, ?, ?, ?, ?, ?, ?)', transactions_to_insert)

        conn.commit(); return True
    except (sqlite3.Error, ValueError, TypeError) as e:
        print(f"!!! VERİTABANI HATASI (Blok {block_num_int}): {e}")
        conn.rollback(); return False
    except Exception as e:
        print(f"!!! Beklenmedik Hata (Blok {block_num_int}): {e}")
        conn.rollback(); return False


# --- ANA VERİ ÇEKME FONKSİYONU ---
def fetch_and_store_range(start_block, end_block, db_path=DB_FILE_PATH):
    """Belirtilen blok aralığını çeker ve veritabanına kaydeder."""
    if not isinstance(start_block, int) or not isinstance(end_block, int) or start_block <= 0 or end_block < start_block:
        print(f"HATA: Geçersiz blok aralığı: {start_block} - {end_block}")
        return

    conn = setup_database(db_path)
    if conn is None:
        print("Veritabanı bağlantısı kurulamadı. İşlem durduruldu.")
        return

    total_blocks = end_block - start_block + 1
    print(f"\nVeri Çekme Başlıyor: {start_block} -> {end_block} ({total_blocks} blok)")
    print(f"Veritabanı: {db_path}")
    print("-" * 40)
    start_time = time.time()
    fetched_count = 0
    error_count = 0

    try:
        for i, block_num in enumerate(range(start_block, end_block + 1)):
            # Alchemy Free Tier limiti saniyede ~333 istek, 1/30 saniye (~33 istek/s) güvenli
            time.sleep(1/30)
            progress = f"[{i+1}/{total_blocks}]"
            print(f"{progress} Blok {block_num} çekiliyor...")
            block_data = get_block_data_robust(block_num)
            if block_data:
                if process_block(conn, block_data):
                    tx_count = len(block_data.get('transactions', []))
                    print(f"{progress} Blok {block_num} işlendi ({tx_count} işlem).")
                    fetched_count += 1
                else:
                    error_count += 1 # DB hatası
            else:
                print(f"{progress} Blok {block_num} çekilemedi, atlanıyor.")
                error_count += 1 # API hatası

    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu.")
    finally:
        if conn:
            conn.close()
            print("Veritabanı bağlantısı kapatıldı.")

        total_time = time.time() - start_time
        print("-" * 40)
        print("Veri Çekme Tamamlandı.")
        print(f"İşlenen Süre: {total_time:.2f} saniye.")
        print(f"Başarıyla Çekilen/İşlenen Blok Sayısı: {fetched_count}")
        print(f"Hata Alınan / Atlanan Blok Sayısı: {error_count}")
        print(f"Veriler '{db_path}' dosyasına kaydedildi.")

if __name__ == "__main__":
    print("Ethereum Veri Çekme Aracı")
    print("-" * 26)
    start_b, end_b = find_block_range_for_date()
    if start_b and end_b:
         fetch_and_store_range(start_b, end_b)
    else:
         print("\nBlok aralığı bulunamadığı için veri çekme başlatılamadı.")
         print("Lütfen config.py dosyasındaki Etherscan API anahtarını kontrol edin.")