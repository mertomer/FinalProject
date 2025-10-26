import sqlite3
import pandas as pd
import os

def get_simulation_pool(start_block, num_blocks):
    """
    Dinamik N bloktaki tüm işlemleri tek bir "havuz" olarak çeken fonksiyon.
    
    Args:
        start_block (int): Başlangıç blok numarası
        num_blocks (int): Çekilecek blok sayısı (N)
    
    Returns:
        tuple: (transactions_df, total_capacity)
            - transactions_df: Pandas DataFrame (txHash, weight, value)
            - total_capacity: N blokluk toplam kapasite (SUM(gasLimit))
    """
    
    # Veritabanı dosyası yolu
    DB_FILE = "ethereum_data.db"
    
    # Veritabanı dosyasının var olup olmadığını kontrol et
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Veritabanı dosyası '{DB_FILE}' bulunamadı. Önce FetchData.py'yi çalıştırın.")
    
    # Veritabanına bağlan
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # N blokluk toplam kapasiteyi çek (SUM(gasLimit))
        cursor.execute('''
            SELECT SUM(gasLimit) as total_capacity
            FROM Blocks 
            WHERE blockNumber >= ? AND blockNumber < ?
        ''', (start_block, start_block + num_blocks))
        
        result = cursor.fetchone()
        total_capacity = result[0] if result[0] is not None else 0
        
        # N bloktaki tüm işlemleri çek (txHash, gas, gasPrice)
        cursor.execute('''
            SELECT txHash, gas, gasPrice
            FROM Transactions 
            WHERE blockNumber >= ? AND blockNumber < ?
            ORDER BY blockNumber, txHash
        ''', (start_block, start_block + num_blocks))
        
        transactions_data = cursor.fetchall()
        
        # Pandas DataFrame oluştur
        df = pd.DataFrame(transactions_data, columns=['txHash', 'gas', 'gasPrice'])
        
        # Kritik düzeltme: Verileri rakama çevir
        # Ağırlık (weight) = int(tx['gas'], 16)
        df['weight'] = df['gas'].apply(lambda x: int(x, 16))
        
        # Ödül (value) = int(tx['gas'], 16) * int(tx['gasPrice'], 16)
        df['value'] = df.apply(lambda row: int(row['gas'], 16) * int(row['gasPrice'], 16), axis=1)
        
        # Sadece gerekli sütunları tut
        transactions_df = df[['txHash', 'weight', 'value']].copy()
        
        print(f"Başarıyla {num_blocks} blok ({start_block} - {start_block + num_blocks - 1}) çekildi:")
        print(f"- Toplam kapasite: {total_capacity:,}")
        print(f"- Toplam işlem sayısı: {len(transactions_df):,}")
        print(f"- Toplam ağırlık: {transactions_df['weight'].sum():,}")
        print(f"- Toplam ödül: {transactions_df['value'].sum():,}")
        
        return transactions_df, total_capacity
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        raise
        
    finally:
        conn.close()

def demo_simulation_pool():
    """
    get_simulation_pool fonksiyonunun nasıl çalışacağını gösteren demo.
    Veritabanı olmadığı için örnek verilerle gösterir.
    """
    print("=" * 60)
    print("get_simulation_pool FONKSİYONU DEMO")
    print("=" * 60)
    
    print("\n📋 Fonksiyon Özellikleri:")
    print("- Veritabanına bağlanır")
    print("- N blokluk toplam kapasiteyi (SUM(gasLimit)) çeker")
    print("- N bloktaki tüm işlemleri (txHash, gas, gasPrice) çeker")
    print("- Ağırlık = int(gas, 16) hesaplar")
    print("- Ödül = int(gas, 16) * int(gasPrice, 16) hesaplar")
    print("- Pandas DataFrame ve total_capacity döndürür")
    
    print("\n🔧 Kullanım Örneği:")
    print("```python")
    print("# 5 blok için havuz oluştur")
    print("transactions_df, total_capacity = get_simulation_pool(20866020, 5)")
    print("")
    print("# 10 blok için havuz oluştur")
    print("transactions_df, total_capacity = get_simulation_pool(20866020, 10)")
    print("")
    print("# 20 blok için havuz oluştur")
    print("transactions_df, total_capacity = get_simulation_pool(20866020, 20)")
    print("```")
    
    print("\n📊 Döndürülen DataFrame Yapısı:")
    print("```")
    print("   txHash                                    weight    value")
    print("0  0x1234...abcd                           21000     4200000000000000")
    print("1  0x5678...efgh                           50000     10000000000000000")
    print("2  0x9abc...ijkl                           100000    20000000000000000")
    print("...")
    print("```")
    
    print("\n⚠️  Önemli Notlar:")
    print("- Veritabanı dosyası (ethereum_data.db) önce oluşturulmalı")
    print("- FetchData.py script'i çalıştırılarak veriler çekilmeli")
    print("- Ağırlık ve ödül hesaplamaları hex'den decimal'e çevrilir")
    print("- Fonksiyon tuple döndürür: (DataFrame, total_capacity)")
    
    print("\n✅ Fonksiyon Hazır!")
    print("Veritabanı oluşturulduktan sonra kullanılabilir.")

if __name__ == "__main__":
    demo_simulation_pool()