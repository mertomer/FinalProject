import sqlite3
import pandas as pd
import os
import random
import sys
import time
import math

# --- YZ SİMÜLASYONU İÇİN GEREKLİ KÜTÜPHANELER ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("HATA: 'deap' kütüphanesi bulunamadı.")
    print("Lütfen terminale 'pip install deap' yazarak kurun.")
    sys.exit(1)

# --- Task 2.1: Veri Havuzu Fonksiyonu ---
def get_simulation_pool(start_block, num_blocks):
    """
    Dinamik N bloktaki tüm işlemleri tek bir "havuz" olarak çeken fonksiyon.
    """
    # Veritabanı dosyasının tam yolunu bul
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DB_FILE = os.path.join(script_dir, "ethereum_data.db")
    
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Veritabanı dosyası '{DB_FILE}' bulunamadı.")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT SUM(gasLimit) FROM Blocks WHERE blockNumber >= ? AND blockNumber < ?', (start_block, start_block + num_blocks))
        total_capacity = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT txHash, gas, gasPrice FROM Transactions WHERE blockNumber >= ? AND blockNumber < ? AND gas IS NOT NULL AND gasPrice IS NOT NULL', (start_block, start_block + num_blocks))
        transactions_data = cursor.fetchall()
        df = pd.DataFrame(transactions_data, columns=['txHash', 'gas', 'gasPrice'])
        
        if df.empty:
            print(f"UYARI: {start_block} ve sonrası için veritabanında işlem bulunamadı.")
            return pd.DataFrame(columns=['txHash', 'weight', 'value']), 0

        df['weight'] = df['gas'].apply(lambda x: int(str(x), 16))
        df['value'] = df.apply(lambda row: int(str(row['gas']), 16) * int(str(row['gasPrice']), 16), axis=1)
        
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

# --- ALGORİTMA FONKSİYONLARI (GÜNCELLENDİ - Yeni Metrikler Eklendi) ---

def calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity):
    """Yardımcı fonksiyon: Ortak metrikleri hesaplar."""
    avg_reward_per_tx = total_value / selected_tx_count if selected_tx_count > 0 else 0
    avg_weight_per_tx = total_weight / selected_tx_count if selected_tx_count > 0 else 0
    cap_util_percent = (total_weight / capacity) * 100 if capacity > 0 else 0
    return avg_reward_per_tx, avg_weight_per_tx, cap_util_percent

def solve_random(transactions_df, capacity):
    """Algoritma 1: Rastgele Seçim (Baseline)"""
    shuffled_df = transactions_df.sample(frac=1)
    total_value, total_weight, selected_tx_count = 0, 0, 0
    
    for _, tx in shuffled_df.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1
            
    avg_reward, avg_weight, cap_util = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Rastgele", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight # YENİ
    }

def solve_greedy(transactions_df, capacity):
    """Algoritma 2: Açgözlü (Greedy) Seçim"""
    df = transactions_df.copy()
    df = df[df['weight'] > 0]
    df['density'] = df['value'] / df['weight']
    df_sorted = df.sort_values(by='density', ascending=False)
    total_value, total_weight, selected_tx_count = 0, 0, 0
    
    for _, tx in df_sorted.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1
            
    avg_reward, avg_weight, cap_util = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Açgözlü (Greedy)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight # YENİ
    }

def solve_genetic_algorithm(transactions_df, capacity):
    """Algoritma 3: Genetik Algoritma (Yapay Zeka)"""
    print("\n[Task 3.2] Genetik Algoritma (YZ) çalıştırılıyor...")
    items = transactions_df[['weight', 'value']].to_dict('records')
    num_items = len(items)

    if not hasattr(creator, "FitnessMax"): creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"): creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_items)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalKnapsack(individual):
        w, v = 0, 0
        for i in range(num_items):
            if individual[i] == 1: w += items[i]['weight']; v += items[i]['value']
        return (v,) if w <= capacity else (0,)

    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", tools.cxTwoPoint); toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)

    start_time = time.time()
    pop = toolbox.population(n=100); hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x)/len(x) if x else 0)
    stats.register("max", lambda x: max(x) if x else 0)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.1, ngen=50, stats=stats, halloffame=hof, verbose=False) # Verbose kapatıldı
    print(f"Genetik Algoritma {time.time() - start_time:.2f} saniyede tamamlandı.")

    best_individual = hof[0]; total_value, total_weight, selected_tx_count = 0, 0, 0
    for i in range(num_items):
        if best_individual[i] == 1:
            total_weight += items[i]['weight']; total_value += items[i]['value']; selected_tx_count += 1
            
    avg_reward, avg_weight, cap_util = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Genetik Algoritma (YZ)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight # YENİ
    }

def solve_simulated_annealing(transactions_df, capacity):
    """Algoritma 4: Benzetilmiş Tavlama (Simulated Annealing - SA)"""
    print("\n[Task 3.3] Benzetilmiş Tavlama (YZ) çalıştırılıyor...")
    items = transactions_df[['weight', 'value']].to_dict('records')
    num_items = len(items)
    
    def get_solution_stats(solution_indices):
        w, v = 0, 0
        for i in solution_indices: w += items[i]['weight']; v += items[i]['value']
        return w, v

    def create_initial_solution():
        sol, w = set(), 0; indices = list(range(num_items)); random.shuffle(indices)
        for i in indices:
            if w + items[i]['weight'] <= capacity: w += items[i]['weight']; sol.add(i)
        _, v = get_solution_stats(sol)
        return sol, w, v

    T_INITIAL, T_MIN, ALPHA = 100.0, 1e-3, 0.99; T = T_INITIAL
    current_solution, current_weight, current_value = create_initial_solution()
    best_solution, best_value, best_weight = current_solution, current_value, current_weight
    
    start_time = time.time()
    while T > T_MIN:
        new_solution = set(current_solution)
        if len(new_solution) > 0 and random.random() < 0.5: new_solution.remove(random.choice(list(new_solution)))
        else:
            item_to_add = random.randint(0, num_items - 1)
            if item_to_add not in new_solution: new_solution.add(item_to_add)

        new_weight, new_value = get_solution_stats(new_solution)
        
        if new_weight <= capacity:
            delta_value = new_value - current_value
            if delta_value > 0 or random.random() < math.exp(delta_value / T):
                current_solution, current_weight, current_value = new_solution, new_weight, new_value
                if current_value > best_value:
                    best_solution, best_weight, best_value = current_solution, new_weight, current_value
        T *= ALPHA
        
    print(f"Benzetilmiş Tavlama {time.time() - start_time:.2f} saniyede tamamlandı.")
    selected_tx_count = len(best_solution)
    avg_reward, avg_weight, cap_util = calculate_derived_metrics(best_value, best_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Benzetilmiş Tavlama (YZ)", "toplam_odul": best_value, "kullanilan_kapasite": best_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight # YENİ
    }


# --- Task 4.1 & 4.2: Ana Simülasyon Çalıştırıcı ve Raporlama (GÜNCELLENDİ) ---
def main_simulation():
    """
    Ana simülasyonu çalıştırır, 4 algoritmayı kıyaslar ve detaylı raporlar.
    """
    print("=" * 60)
    print("ANA SİMÜLASYON BAŞLATILIYOR (Yeni Metrikler Dahil)")
    print("=" * 60)
    
    START_BLOCK = 20866020  
    MIN_WINDOW = 5; MAX_WINDOW = 20
    
    # Yeni sütunlar eklendi
    columns = ['n_blocks', 'algoritma', 'toplam_odul', 
               'kapasite_doluluk_yuzdesi', 'secilen_islem_sayisi', 
               'geride_kalan_islem_sayisi', 
               'ortalama_odul_per_tx', 'ortalama_agirlik_per_tx', # <-- YENİ METRİKLER
               'kullanilan_kapasite', 'toplam_kapasite', 'havuzdaki_islem_sayisi']
    df_results = pd.DataFrame(columns=columns)

    try:
        for n_blocks in range(MIN_WINDOW, MAX_WINDOW + 1):
            print("\n" + "=" * 60 + f"\nPENCERE BOYUTU (N_BLOCKS) = {n_blocks}\n" + "=" * 60)
            
            print(f"\n[Task 2.1] Havuz oluşturuluyor...")
            pool_df, total_capacity = get_simulation_pool(START_BLOCK, n_blocks)
            if pool_df.empty or total_capacity == 0: continue
            havuzdaki_islem_sayisi = len(pool_df)

            # Algoritmaları çalıştır (fonksiyonlar artık yeni metrikleri döndürüyor)
            results_for_n = [
                solve_random(pool_df, total_capacity),
                solve_greedy(pool_df, total_capacity),
                solve_genetic_algorithm(pool_df, total_capacity),
                solve_simulated_annealing(pool_df, total_capacity)
            ]

            print(f"\n--- N={n_blocks} İçin Sonuçlar Kaydediliyor ---")
            for res in results_for_n:
                res['n_blocks'] = n_blocks
                res['toplam_kapasite'] = total_capacity
                res['havuzdaki_islem_sayisi'] = havuzdaki_islem_sayisi
                res['geride_kalan_islem_sayisi'] = havuzdaki_islem_sayisi - res['secilen_islem_sayisi']
                # Yeni metrikler zaten fonksiyonlar tarafından eklendi
                df_results.loc[len(df_results)] = res
        
        # --- Task 4.2: Tüm Sonuçları Raporla ---
        print("\n" + "=" * 60 + "\nTÜM SİMÜLASYONLARIN ÖZET RAPORU (Task 4.2)\n" + "=" * 60)
        
        if df_results.empty: print("Hiçbir sonuç üretilmedi."); return
            
        # Sayısal tipleri onayla
        for col in ['toplam_odul', 'n_blocks', 'kapasite_doluluk_yuzdesi', 'secilen_islem_sayisi', 
                    'geride_kalan_islem_sayisi', 'ortalama_odul_per_tx', 'ortalama_agirlik_per_tx']:
            df_results[col] = pd.to_numeric(df_results[col])

        # --- Kıyaslamalar ---

        # 1. Genel Ortalama Ödül
        print("\n--- ANALİZ 1: ORTALAMA 'BLOK ÖDÜLLERİ' SIRALAMASI ---")
        avg_odul = df_results.groupby('algoritma')['toplam_odul'].mean().sort_values(ascending=False)
        print(avg_odul.to_string(float_format="%.0f"))

        # 2. Ortalama Bekleyen İşlem
        print("\n--- ANALİZ 2: ORTALAMA 'GERİDE KALAN İŞLEM SAYISI' SIRALAMASI ---")
        print("(Düşük = Daha İyi)")
        avg_bekleyen = df_results.groupby('algoritma')['geride_kalan_islem_sayisi'].mean().sort_values(ascending=True)
        print(avg_bekleyen.to_string(float_format="%.0f"))
        
        # YENİ ANALİZLER
        # 3. Ortalama Ödül / İşlem (Kalite)
        print("\n--- ANALİZ 3: ORTALAMA 'ÖDÜL / İŞLEM' SIRALAMASI (Kalite Metriği) ---")
        print("(Yüksek = Daha 'kaliteli' işlemler seçildi)")
        avg_odul_tx = df_results.groupby('algoritma')['ortalama_odul_per_tx'].mean().sort_values(ascending=False)
        print(avg_odul_tx.to_string(float_format="%.0f"))

        # 4. Ortalama Ağırlık / İşlem (Ağırlık Tercihi)
        print("\n--- ANALİZ 4: ORTALAMA 'AĞIRLIK / İŞLEM' SIRALAMASI ---")
        print("(Yüksek = Daha 'ağır' işlemler seçildi)")
        avg_agirlik_tx = df_results.groupby('algoritma')['ortalama_agirlik_per_tx'].mean().sort_values(ascending=False)
        print(avg_agirlik_tx.to_string(float_format="%.0f"))

        # 5. YZ vs Greedy Yüzdesel Kıyaslama
        print("\n--- ANALİZ 5: YZ ALGORİTMALARININ 'AÇGÖZLÜ' YÖNTEME GÖRE ORTALAMA KAZANIMI ---")
        try:
            greedy_avg_odul = avg_odul['Açgözlü (Greedy)']
            if greedy_avg_odul > 0:
                ga_avg_odul = avg_odul.get('Genetik Algoritma (YZ)', 0)
                sa_avg_odul = avg_odul.get('Benzetilmiş Tavlama (YZ)', 0)
                print(f"  Genetik Algoritma (YZ):   % {((ga_avg_odul - greedy_avg_odul) / greedy_avg_odul) * 100:+.2f} daha fazla ödül")
                print(f"  Benzetilmiş Tavlama (YZ): % {((sa_avg_odul - greedy_avg_odul) / greedy_avg_odul) * 100:+.2f} daha fazla ödül")
        except KeyError: print("Greedy sonuçları eksik.")

        # 6. Pencere Büyüdükçe Performans
        print("\n--- ANALİZ 6: PENCERE BÜYÜDÜKÇE PERFORMANS DEĞİŞİMİ (YZ vs Greedy Ödül Yüzdesi) ---")
        try:
            pivot_df = df_results.pivot_table(index='n_blocks', columns='algoritma', values='toplam_odul')
            pivot_df['GA_vs_Greedy_%'] = ((pivot_df['Genetik Algoritma (YZ)'] - pivot_df['Açgözlü (Greedy)']) / pivot_df['Açgözlü (Greedy)']) * 100
            pivot_df['SA_vs_Greedy_%'] = ((pivot_df['Benzetilmiş Tavlama (YZ)'] - pivot_df['Açgözlü (Greedy)']) / pivot_df['Açgözlü (Greedy)']) * 100
            print(pivot_df[['GA_vs_Greedy_%', 'SA_vs_Greedy_%']].to_string(float_format="%.2f%%"))
        except KeyError: print("Kıyaslama için algoritma sonuçlarından biri eksik.")
        except Exception as e: print(f"Pivot tablo hatası: {e}")

        # 7. Raporu Kaydetme
        try:
            rapor_dosyasi = "simulasyon_sonuclari_detayli.csv"
            df_results = df_results[columns] # Sütun sırasını onayla
            df_results.to_csv(rapor_dosyasi, index=False, float_format="%.0f")
            print(f"\n--- TÜM SONUÇLAR BAŞARIYLA '{rapor_dosyasi}' DOSYASINA KAYDEDİLDİ ---")
        except Exception as e: print(f"Rapor dosyası kaydedilemedi: {e}")

    except FileNotFoundError: print("\n!!! HATA: 'ethereum_data.db' bulunamadı.")
    except Exception as e: print(f"\n!!! Simülasyon sırasında hata: {e}")

# --- ANA ÇALIŞTIRMA BLOKU ---
if __name__ == "__main__":
    main_simulation()