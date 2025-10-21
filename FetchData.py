import sqlite3
import requests
import time
import json
import sys

# --- 1. AYARLAR ---
# `find_blocks.py` script'inden aldığınız numaraları buraya yapıştırın.
START_BLOCK = 20866020
END_BLOCK = 20873189

# Alchemy'in RPC adresi
RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/GxHHPQLvsUZ2amJtwrprf"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

DB_FILE = "ethereum_data.db"

def setup_database():
    conn = sqlite3.connect(DB_FILE, timeout=20)  
    cursor = conn.cursor()

    # Blocks Tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Blocks (
        blockNumber INTEGER PRIMARY KEY,
        timestamp INTEGER,
        hash TEXT NOT NULL,
        miner TEXT,
        gasUsed INTEGER,
        gasLimit INTEGER,
        transactionCount INTEGER
    )
    ''')

    # Transactions Tablosu
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Transactions (
        txHash TEXT PRIMARY KEY,
        blockNumber INTEGER,
        fromAddress TEXT,
        toAddress TEXT,
        value_wei TEXT,  -- Hassasiyet kaybı olmasın diye TEXT (string)
        gasPrice TEXT    -- Hassasiyet kaybı olmasın diye TEXT (string)
    )
    ''')
    
    conn.commit()
    print(f"Veritabanı '{DB_FILE}' başarıyla kuruldu/yüklendi.")
    return conn

def get_block_data_robust(block_number, max_retries=5, initial_wait_sec=2):
    """
    Bir blok verisini çeker. API hatası veya rate limit (429) alırsa,
    bekleyerek tekrar dener (exponential backoff).
    """
    hex_block_number = hex(block_number)
    if not hex_block_number.startswith('0x'):
        hex_block_number = '0x' + hex_block_number[2:]
    
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [
            hex_block_number,  
            True              
        ],
        "id": 1
    }
    
    wait_time = initial_wait_sec
    for attempt in range(max_retries):
        try:
            response = requests.post(RPC_URL, json=payload, headers=HEADERS, timeout=15)
            
            response.raise_for_status() 
            
            data = response.json()
            
            if "result" in data and data["result"]:
                return data["result"] 
            else:
                error_msg = data.get('error', 'Bilinmeyen RPC hatası')
                print(f"Blok {block_number}, Deneme {attempt + 1}: RPC Hatası: {error_msg}")
                
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429: 
                print(f"Blok {block_number}: Rate Limit (429) alındı. {wait_time} saniye bekleniyor...")
            else:
                print(f"Blok {block_number}: HTTP Hatası: {http_err}")
        
        except requests.exceptions.RequestException as e:

            print(f"Blok {block_number}: Bağlantı Hatası: {e}")

        if attempt < max_retries - 1:
            time.sleep(wait_time)
            wait_time *= 2 
        
    print(f"HATA: Blok {block_number} {max_retries} deneme sonunda alınamadı.")
    return None

def process_block(conn, block_data):
   
    cursor = conn.cursor()
    
    try:
        # 1. Blok Verisini Kaydet
        cursor.execute('''
        INSERT OR IGNORE INTO Blocks (blockNumber, timestamp, hash, miner, gasUsed, gasLimit, transactionCount)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(block_data['number'], 16),
            int(block_data['timestamp'], 16),
            block_data['hash'],
            block_data['miner'],
            int(block_data['gasUsed'], 16),
            int(block_data['gasLimit'], 16),
            len(block_data['transactions'])
        ))
        
        transactions_to_insert = []
        for tx in block_data['transactions']:
            to_address = tx.get('to') if tx.get('to') else "CONTRACT_CREATION"
            
            transactions_to_insert.append((
                tx['hash'],
                int(tx['blockNumber'], 16),
                tx['from'],
                to_address,
                tx['value'],    
                tx['gasPrice']  
            ))
        
        cursor.executemany('''
        INSERT OR IGNORE INTO Transactions (txHash, blockNumber, fromAddress, toAddress, value_wei, gasPrice)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', transactions_to_insert)

        conn.commit() # İşlemleri onayla
        return True
        
    except Exception as e:
        print(f"!!! VERİTABANI HATASI (Blok {int(block_data['number'], 16)}): {e}")
        conn.rollback()
        return False

def main():
    if START_BLOCK == 0 or END_BLOCK == 0:
        print("HATA: Lütfen `fetch_data.py` dosyasını açın.")
        print("`START_BLOCK` ve `END_BLOCK` değişkenlerini `find_blocks.py` script'inden aldığınız değerlerle güncelleyin.")
        sys.exit(1) 

    conn = setup_database()
    
    total_blocks = END_BLOCK - START_BLOCK + 1
    print(f"İşlem başlıyor: {START_BLOCK}'dan {END_BLOCK}'a kadar {total_blocks} blok çekilecek...")
    print("Bu işlem ağ yoğunluğuna ve rate limitlere bağlı olarak saatler sürebilir.")
    print("Konsolda 'Rate Limit (429)' görmek normaldir.")
    print("-" * 40)
    
    start_time = time.time()

    for i, block_num in enumerate(range(START_BLOCK, END_BLOCK + 1)):
        time.sleep(1/30)  
        progress = f"[{i+1}/{total_blocks}]" 
        
        print(f"{progress} Blok {block_num} çekiliyor...")
        
        block_data = get_block_data_robust(block_num)
        
        if block_data:
            success = process_block(conn, block_data)
            if success:
                tx_count = len(block_data.get('transactions', []))
                print(f"{progress} Blok {block_num} veritabanına işlendi. ({tx_count} işlem)")
        else:
            print(f"{progress} Blok {block_num} çekilemedi, atlanıyor.")

    conn.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    print("-" * 40)
    print("İşlem tamamlandı.")
    print(f"Toplam {total_blocks} blok {total_time:.2f} saniyede işlendi.")
    print(f"Tüm verileriniz '{DB_FILE}' dosyasında.")

if __name__ == "__main__":
    main()