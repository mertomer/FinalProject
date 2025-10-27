# eth_analyzer/simulation/core.py
import sqlite3
import pandas as pd
import os
# config'den DB yolunu import et
from ..config import DB_FILE_PATH

def get_simulation_pool(start_block, num_blocks):
    """
    Dinamik N bloktaki tüm işlemleri tek bir "havuz" olarak çeken fonksiyon.
    Veritabanı yolu config dosyasından alınır.
    """
    if not os.path.exists(DB_FILE_PATH):
        raise FileNotFoundError(f"Veritabanı dosyası '{DB_FILE_PATH}' bulunamadı. Önce veri çekme script'ini çalıştırın.")

    conn = sqlite3.connect(DB_FILE_PATH)
    cursor = conn.cursor()
    end_block_exclusive = start_block + num_blocks # SQL BETWEEN yerine >= ve < kullanmak daha nettir

    try:
        # N blokluk toplam kapasiteyi çek
        cursor.execute('SELECT SUM(gasLimit) FROM Blocks WHERE blockNumber >= ? AND blockNumber < ?',
                       (start_block, end_block_exclusive))
        result = cursor.fetchone()
        total_capacity = result[0] if result and result[0] is not None else 0

        # N bloktaki tüm işlemleri çek
        cursor.execute('''
            SELECT txHash, gas, gasPrice FROM Transactions
            WHERE blockNumber >= ? AND blockNumber < ?
            AND gas IS NOT NULL AND gasPrice IS NOT NULL
            AND gas != '0x0' AND gasPrice != '0x0'
        ''', (start_block, end_block_exclusive))
        transactions_data = cursor.fetchall()
        df = pd.DataFrame(transactions_data, columns=['txHash', 'gas', 'gasPrice'])

        if df.empty:
            print(f"UYARI: {start_block} - {end_block_exclusive - 1} aralığı için veritabanında geçerli işlem bulunamadı.")
            return pd.DataFrame(columns=['txHash', 'weight', 'value']), 0

        # Verileri rakama çevir (Hata kontrolü eklendi)
        valid_rows = []
        for index, row in df.iterrows():
            try:
                weight = int(str(row['gas']), 16)
                price = int(str(row['gasPrice']), 16)
                if weight > 0 and price > 0: # Sadece geçerli ağırlık ve fiyata sahip olanları al
                    value = weight * price
                    valid_rows.append({'txHash': row['txHash'], 'weight': weight, 'value': value})
            except (ValueError, TypeError):
                # Geçersiz hex değeri varsa atla
                print(f"Uyarı: Geçersiz hex değeri atlanıyor (tx: {row['txHash']}) gas: {row['gas']}, gasPrice: {row['gasPrice']}")
                continue

        if not valid_rows:
             print(f"UYARI: {start_block} - {end_block_exclusive - 1} aralığında geçerli işlem metriği hesaplanamadı.")
             return pd.DataFrame(columns=['txHash', 'weight', 'value']), 0

        transactions_df = pd.DataFrame(valid_rows)

        print(f"Başarıyla {num_blocks} blok ({start_block} - {end_block_exclusive - 1}) havuzu oluşturuldu:")
        print(f"- Toplam kapasite (Çanta Büyüklüğü): {total_capacity:,}")
        print(f"- Havuzdaki geçerli işlem sayısı (Eşyalar): {len(transactions_df):,}")

        return transactions_df, total_capacity

    except sqlite3.Error as e:
        print(f"get_simulation_pool sırasında veritabanı hatası: {e}")
        raise # Hatanın yukarıya bildirilmesi önemli
    except Exception as e:
        print(f"get_simulation_pool sırasında beklenmedik hata: {e}")
        raise
    finally:
        if conn:
            conn.close()