import sqlite3
import pandas as pd
import os
import random
import sys
import time
import math  # Benzetilmiş Tavlama (SA) için eklendi

# --- YZ SİMÜLASYONU İÇİN GEREKLİ KÜTÜPHANELER (Task 1.2) ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("HATA: 'deap' kütüphanesi bulunamadı.")
    print("Lütfen terminale 'pip install deap' yazarak kurun.")
    sys.exit(1)


# --- Task 2.1: Veri Havuzu Fonksiyonu (Tamamlandı) ---
def get_simulation_pool(start_block, num_blocks):
    """
    Dinamik N bloktaki tüm işlemleri tek bir "havuz" olarak çeken fonksiyon.
    """
    
    # Veritabanı dosyasının tam yolunu bul
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DB_FILE = os.path.join(script_dir, "ethereum_data.db")
    
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Veritabanı dosyası '{DB_FILE}' bulunamadı. Önce FetchData.py'yi çalıştırın.")
    
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
            AND gas IS NOT NULL AND gasPrice IS NOT NULL
        ''', (start_block, start_block + num_blocks))
        
        transactions_data = cursor.fetchall()
        df = pd.DataFrame(transactions_data, columns=['txHash', 'gas', 'gasPrice'])
        
        if df.empty:
            print(f"UYARI: {start_block} ve sonrası için veritabanında işlem bulunamadı.")
            return pd.DataFrame(columns=['txHash', 'weight', 'value']), 0

        # Kritik düzeltme: Verileri rakama çevir
        df['weight'] = df['gas'].apply(lambda x: int(str(x), 16))
        df['value'] = df.apply(lambda row: int(str(row['gas']), 16) * int(str(row['gasPrice']), 16), axis=1)
        
        # Sadece gerekli sütunları tut ve "0" ağırlıklı/değerli işlemleri temizle
        transactions_df = df[['txHash', 'weight', 'value']].copy()
        transactions_df = transactions_df[(transactions_df['weight'] > 0) & (transactions_df['value'] > 0)]
        
        print(f"Başarıyla {num_blocks} blok ({start_block} - {start_block + num_blocks - 1}) çekildi:")
        print(f"- Toplam kapasite (Çanta Büyüklüğü): {total_capacity:,}")
        print(f"- Toplam işlem sayısı (Eşyalar): {len(transactions_df):,}")
        
        return transactions_df, total_capacity
        
    except Exception as e:
        print(f"get_simulation_pool sırasında Hata oluştu: {e}")
        raise
    finally:
        conn.close()

# --- Task 2.2: Algoritma 1: Rastgele Seçim (Tamamlandı) ---
def solve_random(transactions_df, capacity):
    """
    Algoritma 1: Rastgele Seçim (Baseline)
    """
    shuffled_df = transactions_df.sample(frac=1)
    
    total_value = 0
    total_weight = 0
    selected_tx_count = 0
    
    for _, tx in shuffled_df.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1
            
    result = {
        "algoritma": "Rastgele",
        "toplam_odul": total_value,
        "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count,
        "kapasite_doluluk_yuzdesi": (total_weight / capacity) * 100 if capacity > 0 else 0
    }
    return result

# --- Task 3.1: Algoritma 2: Açgözlü (Greedy) Seçim (Tamamlandı) ---
def solve_greedy(transactions_df, capacity):
    """
    Algoritma 2: Açgözlü (Greedy) Seçim
    Birim ağırlık başına en yüksek ödülü (value/weight ratio) 
    veren işlemleri öncelikli olarak seçer.
    """
    df = transactions_df.copy()
    df = df[df['weight'] > 0]
    df['density'] = df['value'] / df['weight']
    df_sorted = df.sort_values(by='density', ascending=False)
    
    total_value = 0
    total_weight = 0
    selected_tx_count = 0
    
    for _, tx in df_sorted.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1
            
    result = {
        "algoritma": "Açgözlü (Greedy)",
        "toplam_odul": total_value,
        "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count,
        "kapasite_doluluk_yuzdesi": (total_weight / capacity) * 100 if capacity > 0 else 0
    }
    return result

# --- Task 3.2: Algoritma 3: Genetik Algoritma (YZ) (Tamamlandı) ---
def solve_genetic_algorithm(transactions_df, capacity):
    """
    Algoritma 3: Genetik Algoritma (Yapay Zeka)
    DEAP kütüphanesi kullanarak optimum çözümü bulmaya çalışır.
    """
    print("\n[Task 3.2] Genetik Algoritma (YZ) çalıştırılıyor...")
    print("Popülasyon oluşturuluyor ve evrim süreci başlıyor (Bu işlem biraz sürebilir)...")
    
    items = transactions_df[['weight', 'value']].to_dict('records')
    num_items = len(items)

    # --- GA Kurulumu (DEAP) ---
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_items)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalKnapsack(individual):
        total_weight = 0
        total_value = 0
        for i in range(num_items):
            if individual[i] == 1: 
                total_weight += items[i]['weight']
                total_value += items[i]['value']
        
        if total_weight > capacity:
            return (0,)
        else:
            return (total_value,)

    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)

    start_time = time.time()
    pop = toolbox.population(n=100) 
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x) / len(x) if x else 0)
    stats.register("std", lambda x: (sum((i - sum(x)/len(x))**2 for i in x) / len(x))**0.5 if x else 0)
    stats.register("min", lambda x: min(x) if x else 0)
    stats.register("max", lambda x: max(x) if x else 0)

    algorithms.eaSimple(pop, toolbox, 
                        cxpb=0.7, 
                        mutpb=0.1, 
                        ngen=50, 
                        stats=stats, 
                        halloffame=hof, 
                        verbose=True)

    end_time = time.time()
    print(f"Genetik Algoritma {end_time - start_time:.2f} saniyede tamamlandı.")

    best_individual = hof[0]
    
    total_value = 0
    total_weight = 0
    selected_tx_count = 0
    for i in range(num_items):
        if best_individual[i] == 1:
            total_weight += items[i]['weight']
            total_value += items[i]['value']
            selected_tx_count += 1

    result = {
        "algoritma": "Genetik Algoritma (YZ)",
        "toplam_odul": total_value,
        "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count,
        "kapasite_doluluk_yuzdesi": (total_weight / capacity) * 100 if capacity > 0 else 0
    }
    return result

# --- YENİ EKLENDİ ---
# --- Task 3.3 (Alternatif): Algoritma 4: Benzetilmiş Tavlama (YZ) ---
def solve_simulated_annealing(transactions_df, capacity):
    """
    Algoritma 4: Benzetilmiş Tavlama (Simulated Annealing - SA)
    Farklı bir YZ/metaheuristic yaklaşımı.
    """
    print("\n[Task 3.3] Benzetilmiş Tavlama (YZ) çalıştırılıyor...")
    print("Sistem 'ısıtılıyor' ve yavaşça 'soğutuluyor' (Bu işlem biraz sürebilir)...")
    
    # DataFrame'i daha hızlı erişim için listeye çevir
    items = transactions_df[['weight', 'value']].to_dict('records')
    num_items = len(items)
    
    # --- Yardımcı Fonksiyonlar ---
    def get_solution_stats(solution_indices):
        """Verilen bir çözümün (indeks listesi) ağırlık ve değerini hesaplar."""
        weight = 0
        value = 0
        for i in solution_indices:
            weight += items[i]['weight']
            value += items[i]['value']
        return weight, value

    def create_initial_solution():
        """Kapasiteyi aşmayan rastgele bir başlangıç çözümü oluşturur."""
        current_solution = set() # Hızlı ekleme/çıkarma için 'set' kullan
        current_weight = 0
        
        # Rastgele sırayla işlemlere bak
        indices = list(range(num_items))
        random.shuffle(indices)
        
        for i in indices:
            if current_weight + items[i]['weight'] <= capacity:
                current_weight += items[i]['weight']
                current_solution.add(i)
        
        _, current_value = get_solution_stats(current_solution)
        return current_solution, current_weight, current_value

    # --- SA Parametreleri ---
    T_INITIAL = 100.0   # Başlangıç sıcaklığı
    T_MIN = 1e-3        # Bitiş sıcaklığı
    ALPHA = 0.99        # Soğutma oranı
    
    T = T_INITIAL
    
    # Başlangıç çözümünü oluştur
    current_solution, current_weight, current_value = create_initial_solution()
    
    # Bulunan en iyi çözümü sakla
    best_solution = current_solution
    best_value = current_value
    best_weight = current_weight
    
    start_time = time.time()
    
    # Sistem "donana" kadar döngüye gir
    while T > T_MIN:
        # Rastgele bir "komşu" çözüm oluştur
        
        # 1. Küçük bir değişiklik yap (örn: birini çıkar, birini ekle)
        new_solution = set(current_solution) # Kopyala
        
        # %50 olasılıkla birini çıkar (eğer boş değilse)
        if len(new_solution) > 0 and random.random() < 0.5:
            item_to_remove = random.choice(list(new_solution))
            new_solution.remove(item_to_remove)
        
        # %50 olasılıkla birini ekle (eğer zaten ekli değilse)
        else:
            item_to_add = random.randint(0, num_items - 1)
            if item_to_add not in new_solution:
                new_solution.add(item_to_add)

        # 2. Yeni çözümü değerlendir
        new_weight, new_value = get_solution_stats(new_solution)
        
        # 3. Karar ver
        if new_weight <= capacity: # Çözüm geçerli mi?
            # Enerji farkı (ΔE)
            delta_value = new_value - current_value
            
            if delta_value > 0:
                # Yeni çözüm daha iyi -> Her zaman kabul et
                current_solution, current_weight, current_value = new_solution, new_weight, new_value
                # En iyiyi güncelle
                if current_value > best_value:
                    best_solution, best_weight, best_value = current_solution, new_weight, current_value
            else:
                # Yeni çözüm daha kötü -> Olasılıkla kabul et
                # exp(ΔE / T)
                acceptance_prob = math.exp(delta_value / T)
                if random.random() < acceptance_prob:
                    current_solution, current_weight, current_value = new_solution, new_weight, new_value
                    
        # 4. Sıcaklığı düşür (Soğutma)
        T *= ALPHA
        
    end_time = time.time()
    print(f"Benzetilmiş Tavlama {end_time - start_time:.2f} saniyede tamamlandı.")

    result = {
        "algoritma": "Benzetilmiş Tavlama (YZ)",
        "toplam_odul": best_value,
        "kullanilan_kapasite": best_weight,
        "secilen_islem_sayisi": len(best_solution),
        "kapasite_doluluk_yuzdesi": (best_weight / capacity) * 100 if capacity > 0 else 0
    }
    return result

# --- Task 4.1: Ana Simülasyon Çalıştırıcı (GÜNCELLENDİ) ---
# --- Task 4.1 & 4.2: Ana Simülasyon Çalıştırıcı (GÜNCELLENDİ) ---
# --- Task 4.1 & 4.2: Ana Simülasyon Çalıştırıcı ve Raporlama ---
def main_simulation():
    """
    Ana simülasyonu çalıştırır.
    Hocanın isteğine göre "büyüyen pencere" (growing window) analizi yapar.
    Örn: 5 blokluk havuzdan 20 blokluk havuza kadar tüm senaryoları dener
    ve 4 algoritmayı da kıyaslar.
    """
    print("=" * 60)
    print("ANA SİMÜLASYON BAŞLATILIYOR (Büyüyen Pencere Analizi)")
    print("=" * 60)
    
    # --- Simülasyon Ayarları ---
    # Bu numarayı veritabanınızdaki bir başlangıç bloğu ile değiştirebilirsiniz
    START_BLOCK = 20866020  
    
    # "1-5", "1-6", ... "1-20" arası senaryolar için:
    MIN_WINDOW = 5   # Başlangıç pencere boyutu (örn: 5 blok)
    MAX_WINDOW = 20  # Bitiş pencere boyutu (örn: 20 blok)
    
    # Tüm sonuçları saklamak için bir liste
    all_results = []
    
    # Raporlamayı kolaylaştırmak için ana DataFrame'i baştan oluştur
    columns = ['n_blocks', 'algoritma', 'toplam_odul', 
               'kapasite_doluluk_yuzdesi', 'secilen_islem_sayisi', 
               'kullanilan_kapasite', 'toplam_kapasite', 'havuzdaki_islem_sayisi']
    df_results = pd.DataFrame(columns=columns)

    try:
        # Büyüyen pencere döngüsü (Task 4.1'in başlangıcı)
        for n_blocks in range(MIN_WINDOW, MAX_WINDOW + 1):
            print("\n" + "=" * 60)
            print(f"PENCERE BOYUTU (N_BLOCKS) = {n_blocks} İÇİN ÇALIŞTIRILIYOR")
            print("=" * 60)
            
            # --- Task 2.1: Havuzu oluştur ---
            print(f"\n[Task 2.1] {n_blocks} blok için havuz oluşturuluyor...")
            pool_df, total_capacity = get_simulation_pool(START_BLOCK, n_blocks)
            
            if pool_df.empty or total_capacity == 0:
                print("Havuzda hiç işlem bulunamadı. Bu pencere atlanıyor.")
                continue

            havuzdaki_islem_sayisi = len(pool_df)
            print(f"[Task 2.1] Havuz ve kapasite başarıyla alındı.")

            # --- Algoritmaları Çalıştır ---
            print(f"\n[Task 2.2] Rastgele Seçim çalıştırılıyor...")
            random_result = solve_random(pool_df, total_capacity)
            
            print(f"\n[Task 3.1] Açgözlü (Greedy) çalıştırılıyor...")
            greedy_result = solve_greedy(pool_df, total_capacity)

            print(f"\n[Task 3.2] Genetik Algoritma (YZ) çalıştırılıyor...")
            # Not: GA'nın "verbose=True" çıktısını kapatmak için 
            # solve_genetic_algorithm içindeki algorithms.eaSimple'da verbose=False yapabilirsiniz
            ga_result = solve_genetic_algorithm(pool_df, total_capacity)
            
            print(f"\n[Task 3.3] Benzetilmiş Tavlama (YZ) çalıştırılıyor...")
            sa_result = solve_simulated_annealing(pool_df, total_capacity)

            # Sonuçları kaydet
            print(f"\n--- N={n_blocks} İçin Sonuçlar Kaydediliyor ---")
            for res in [random_result, greedy_result, ga_result, sa_result]:
                res['n_blocks'] = n_blocks
                res['toplam_kapasite'] = total_capacity
                res['havuzdaki_islem_sayisi'] = havuzdaki_islem_sayisi
                # DataFrame'e satır olarak ekle
                df_results.loc[len(df_results)] = res
        
        # --- Task 4.2: Tüm Sonuçları Raporla ---
        print("\n" + "=" * 60)
        print("TÜM SİMÜLASYONLARIN ÖZET RAPORU (Task 4.2)")
        print("=" * 60)
        
        if df_results.empty:
            print("Hiçbir sonuç üretilmedi. Veritabanını kontrol edin.")
            return
            
        # Raporlama için sayısal tipleri onayla
        df_results['toplam_odul'] = pd.to_numeric(df_results['toplam_odul'])
        df_results['n_blocks'] = pd.to_numeric(df_results['n_blocks'])

        # --- Hocanızın İstediği Kıyaslamalar ---

        # 1. Analiz: Algoritma Bazında Ortalama Performans
        print("\n--- ANALİZ 1: ALGORİTMA BAZINDA ORTALAMA PERFORMANS ÖZETİ ---")
        avg_performance = df_results.groupby('algoritma')['toplam_odul'].mean().sort_values(ascending=False)
        print(avg_performance.to_string(float_format="%.0f")) # Ondalıksız göster
        
        # Baseline (Greedy) ödülünü al
        greedy_avg_odul = avg_performance.get('Açgözlü (Greedy)', 0)

        if greedy_avg_odul > 0:
            print("\n--- ANALİZ 2: YZ ALGORİTMALARININ 'AÇGÖZLÜ' YÖNTEME GÖRE ORTALAMA KAZANIMI ---")
            ga_avg_odul = avg_performance.get('Genetik Algoritma (YZ)', 0)
            sa_avg_odul = avg_performance.get('Benzetilmiş Tavlama (YZ)', 0)
            
            ga_kazanc = ((ga_avg_odul - greedy_avg_odul) / greedy_avg_odul) * 100
            sa_kazanc = ((sa_avg_odul - greedy_avg_odul) / greedy_avg_odul) * 100
            
            print(f"  Genetik Algoritma (YZ):   % {ga_kazanc:+.2f} daha fazla ödül")
            print(f"  Benzetilmiş Tavlama (YZ): % {sa_kazanc:+.2f} daha fazla ödül")

        # 2. Analiz: Pencere Boyutunun Performansa Etkisi (En Önemli Analiz)
        print("\n--- ANALİZ 3: PENCERE BOYUTU BÜYÜDÜKÇE PERFORMANS DEĞİŞİMİ (Greedy vs GA) ---")
        
        # Pivot tablo oluştur: Her 'n_blocks' için 'algoritma'ların 'toplam_odul'u
        pivot_df = df_results.pivot_table(index='n_blocks', columns='algoritma', values='toplam_odul')
        
        # YZ'nin Greedy'ye göre yüzdesel farkını hesapla
        if 'Açgözlü (Greedy)' in pivot_df.columns and 'Genetik Algoritma (YZ)' in pivot_df.columns:
            pivot_df['GA_vs_Greedy_Yuzde'] = ((pivot_df['Genetik Algoritma (YZ)'] - pivot_df['Açgözlü (Greedy)']) / pivot_df['Açgözlü (Greedy)']) * 100
            print(pivot_df[['GA_vs_Greedy_Yuzde']].to_string(float_format="%.2f%%"))
        else:
            print("Kıyaslama için 'Açgözlü (Greedy)' veya 'Genetik Algoritma (YZ)' sonuçları eksik.")

        # Bu DataFrame'i bir Excel/CSV dosyasına kaydedebilirsiniz
        try:
            rapor_dosyasi = "simulasyon_sonuclari.csv"
            df_results.to_csv(rapor_dosyasi, index=False, float_format="%.0f")
            print(f"\n--- TÜM SONUÇLAR BAŞARIYLA '{rapor_dosyasi}' DOSYASINA KAYDEDİLDİ ---")
        except Exception as e:
            print(f"Rapor dosyası kaydedilemedi: {e}")

        
    except FileNotFoundError:
        print("\n!!! HATA: 'ethereum_data.db' bulunamadı.")
        print("Lütfen önce veritabanını (FetchData.py ile) oluşturduğunuzdan emin olun.")
    except Exception as e:
        print(f"\n!!! Simülasyon sırasında beklenmedik bir hata oluştu: {e}")


# --- ANA ÇALIŞTIRMA BLOKU ---
if __name__ == "__main__":
    main_simulation()