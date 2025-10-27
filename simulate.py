# simulate.py (Ana Çalıştırıcı - Kök Dizin)
import pandas as pd
import numpy as np
import sys
import os
import time  # time modülünü ekleyelim

# Proje kök dizinini Python yoluna ekle (importların çalışması için)
project_root = os.path.dirname(os.path.abspath(__file__))
# Eğer eth_analyzer klasörü kök dizindeyse sys.path eklemeye gerek yok
# Eğer değilse (örn: src/eth_analyzer ise) aşağıdaki satır gerekir:
# sys.path.insert(0, os.path.join(project_root, 'src')) # 'src' yerine sizin klasör adınız

# eth_analyzer paketinden gerekli modülleri ve ayarları import et
try:
    from eth_analyzer.config import (
        DEFAULT_START_BLOCK, SIMULATION_MIN_WINDOW, SIMULATION_MAX_WINDOW,
        REPORT_CSV_FILE # Rapor dosyasının tam yolu config'de
    )
    from eth_analyzer.simulation.core import get_simulation_pool
    from eth_analyzer.simulation.algorithms import (
        solve_random, solve_greedy,
        solve_genetic_algorithm, solve_simulated_annealing
    )
except ImportError as e:
    print(f"HATA: Gerekli modüller import edilemedi: {e}")
    print("Proje yapısını ve __init__.py dosyalarını kontrol edin.")
    sys.exit(1)

def main_simulation(start_block=DEFAULT_START_BLOCK,
                    min_window=SIMULATION_MIN_WINDOW,
                    max_window=SIMULATION_MAX_WINDOW,
                    report_file=REPORT_CSV_FILE):
    """
    Ana simülasyonu çalıştırır, 4 algoritmayı kıyaslar ve detaylı raporlar.
    """
    print("=" * 60)
    print("ANA SİMÜLASYON BAŞLATILIYOR (Modüler Yapı)")
    print("=" * 60)

    # Rapor dosyasının kaydedileceği dizini kontrol et/oluştur
    results_dir = os.path.dirname(report_file)
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"'{results_dir}' dizini oluşturuldu.")
    except OSError as e:
        print(f"HATA: Sonuç dizini oluşturulamadı: {e}")
        return # Dizini oluşturamazsak devam etmenin anlamı yok

    columns = ['n_blocks', 'algoritma', 'toplam_odul',
               'kapasite_doluluk_yuzdesi', 'secilen_islem_sayisi',
               'geride_kalan_islem_sayisi',
               'ortalama_odul_per_tx', 'ortalama_agirlik_per_tx',
               'kullanilan_kapasite', 'toplam_kapasite', 'havuzdaki_islem_sayisi']
    df_results = pd.DataFrame(columns=columns)

    start_run_time = time.time() # Tüm simülasyonun süresini ölç

    try:
        for n_blocks in range(min_window, max_window + 1):
            print("\n" + "=" * 60 + f"\nPENCERE BOYUTU = {n_blocks}\n" + "=" * 60)

            print(f"\n[Core] Havuz oluşturuluyor...")
            pool_df, total_capacity = get_simulation_pool(start_block, n_blocks)
            if pool_df is None or total_capacity is None: # Hata durumunda None dönebilir
                 print("Havuz oluşturulamadı, bu pencere atlanıyor.")
                 continue
            if pool_df.empty or total_capacity <= 0:
                 print("Havuz boş veya kapasite geçersiz, bu pencere atlanıyor.")
                 continue
            havuzdaki_islem_sayisi = len(pool_df)

            print(f"\n[Algorithms] Algoritmalar çalıştırılıyor...")
            # Algoritmaları tuple listesi olarak tanımla (daha modüler)
            algorithms_to_run = [
                ("Rastgele", solve_random),
                ("Açgözlü (Greedy)", solve_greedy),
                ("Genetik Algoritma (YZ)", solve_genetic_algorithm),
                ("Benzetilmiş Tavlama (YZ)", solve_simulated_annealing)
            ]

            results_for_n = []
            for algo_name, algo_func in algorithms_to_run:
                print(f"\n--- Çalıştırılıyor: {algo_name} ---")
                try:
                    res = algo_func(pool_df, total_capacity)
                    res['n_blocks'] = n_blocks
                    res['toplam_kapasite'] = total_capacity
                    res['havuzdaki_islem_sayisi'] = havuzdaki_islem_sayisi
                    res['geride_kalan_islem_sayisi'] = havuzdaki_islem_sayisi - res.get('secilen_islem_sayisi', 0)
                    results_for_n.append(res)
                except Exception as e:
                    print(f"HATA: '{algo_name}' algoritması çalıştırılırken hata oluştu: {e}")
                    # Hata durumunda bile devam etmesi için boş sonuç ekleyebiliriz
                    # results_for_n.append({'algoritma': algo_name, 'n_blocks': n_blocks, 'toplam_odul': None, ...})

            print(f"\n--- N={n_blocks} İçin Sonuçlar Kaydediliyor ---")
            # DataFrame'e eklemeden önce eksik sütunları doldur (hata durumları için)
            temp_df = pd.DataFrame(results_for_n)
            for col in columns:
                if col not in temp_df.columns:
                    temp_df[col] = None # Veya pd.NA
            df_results = pd.concat([df_results, temp_df[columns]], ignore_index=True)


        # --- Raporlama ---
        print("\n" + "=" * 60 + "\nÖZET RAPOR\n" + "=" * 60)
        if df_results.empty: print("Hiçbir sonuç üretilmedi."); return

        # Sayısal tipleri onayla (NaN'ları 0 yapabiliriz)
        for col in ['toplam_odul', 'kapasite_doluluk_yuzdesi', 'secilen_islem_sayisi',
                    'geride_kalan_islem_sayisi', 'ortalama_odul_per_tx', 'ortalama_agirlik_per_tx',
                    'kullanilan_kapasite']:
            df_results[col] = pd.to_numeric(df_results[col], errors='coerce').fillna(0)
        df_results['n_blocks'] = pd.to_numeric(df_results['n_blocks'])

        # --- Kıyaslamalar ---
        # (Analiz 1 - Ortalama Ödül)
        print("\n--- ANALİZ 1: ORTALAMA 'BLOK ÖDÜLLERİ' ---")
        avg_odul = df_results.groupby('algoritma')['toplam_odul'].mean().sort_values(ascending=False)
        print(avg_odul.to_string(float_format="%.0f"))

        # (Analiz 2 - Ortalama Bekleyen İşlem)
        print("\n--- ANALİZ 2: ORTALAMA 'GERİDE KALAN İŞLEM SAYISI' ---")
        avg_bekleyen = df_results.groupby('algoritma')['geride_kalan_islem_sayisi'].mean().sort_values(ascending=True)
        print(avg_bekleyen.to_string(float_format="%.0f"))

        # (Analiz 3 - Ortalama Ödül / İşlem)
        print("\n--- ANALİZ 3: ORTALAMA 'ÖDÜL / İŞLEM' (Kalite) ---")
        avg_odul_tx = df_results.groupby('algoritma')['ortalama_odul_per_tx'].mean().sort_values(ascending=False)
        print(avg_odul_tx.to_string(float_format="%.0f"))

        # (Analiz 4 - Ortalama Ağırlık / İşlem)
        print("\n--- ANALİZ 4: ORTALAMA 'AĞIRLIK / İŞLEM' ---")
        avg_agirlik_tx = df_results.groupby('algoritma')['ortalama_agirlik_per_tx'].mean().sort_values(ascending=False)
        print(avg_agirlik_tx.to_string(float_format="%.0f"))

        # (Analiz 5 - YZ vs Greedy Yüzdesel)
        print("\n--- ANALİZ 5: YZ vs GREEDY ORTALAMA KAZANIM (%) ---")
        # ... (Eski koddaki YZ vs Greedy yüzdesel hesaplama bloğu buraya) ...
        try:
            greedy_avg_odul = avg_odul.loc['Açgözlü (Greedy)']
            if greedy_avg_odul > 0:
                ga_avg_odul = avg_odul.get('Genetik Algoritma (YZ)', 0)
                sa_avg_odul = avg_odul.get('Benzetilmiş Tavlama (YZ)', 0)
                print(f"  GA vs Greedy (Ödül):   % {((ga_avg_odul - greedy_avg_odul) / greedy_avg_odul) * 100:+.2f}")
                print(f"  SA vs Greedy (Ödül):   % {((sa_avg_odul - greedy_avg_odul) / greedy_avg_odul) * 100:+.2f}")
            else: print("Greedy ödülü 0 olduğu için yüzdesel kıyas yapılamadı.")

            greedy_avg_bekleyen = avg_bekleyen.loc['Açgözlü (Greedy)']
            if greedy_avg_bekleyen > 0:
                 ga_avg_bekleyen = avg_bekleyen.get('Genetik Algoritma (YZ)', float('inf'))
                 sa_avg_bekleyen = avg_bekleyen.get('Benzetilmiş Tavlama (YZ)', float('inf'))
                 print(f"  GA vs Greedy (Bekleme): % {-((ga_avg_bekleyen - greedy_avg_bekleyen) / greedy_avg_bekleyen) * 100:+.2f}") # (-) iyileşme
                 print(f"  SA vs Greedy (Bekleme): % {-((sa_avg_bekleyen - greedy_avg_bekleyen) / greedy_avg_bekleyen) * 100:+.2f}") # (-) iyileşme
            else: print("Greedy bekleyen işlem sayısı 0 olduğu için yüzdesel kıyas yapılamadı.")

        except KeyError: print("Kıyaslama için 'Açgözlü (Greedy)' sonucu eksik.")
        except ZeroDivisionError: print("Yüzdesel kıyaslamada sıfıra bölme hatası.")


        # (Analiz 6 - Pencere Büyüdükçe Performans)
        print("\n--- ANALİZ 6: PENCERE BÜYÜDÜKÇE (N_BLOCKS) YZ vs GREEDY ÖDÜL FARKI (%) ---")
        # ... (Eski koddaki pivot tablo bloğu buraya) ...
        try:
            pivot_df = df_results.pivot_table(index='n_blocks', columns='algoritma', values='toplam_odul')
            # Pivot tablo oluşurken NaN gelmemesi için kontrol
            required_algos = ['Açgözlü (Greedy)', 'Genetik Algoritma (YZ)', 'Benzetilmiş Tavlama (YZ)']
            if all(algo in pivot_df.columns for algo in required_algos):
                # Sıfıra bölmeyi engelle
                pivot_df['GA_vs_Greedy_%'] = np.where(pivot_df['Açgözlü (Greedy)'] != 0, ((pivot_df['Genetik Algoritma (YZ)'] - pivot_df['Açgözlü (Greedy)']) / pivot_df['Açgözlü (Greedy)']) * 100, 0)
                pivot_df['SA_vs_Greedy_%'] = np.where(pivot_df['Açgözlü (Greedy)'] != 0, ((pivot_df['Benzetilmiş Tavlama (YZ)'] - pivot_df['Açgözlü (Greedy)']) / pivot_df['Açgözlü (Greedy)']) * 100, 0)
                print(pivot_df[['GA_vs_Greedy_%', 'SA_vs_Greedy_%']].to_string(float_format="%.2f%%"))
            else: print("Pivot tablo için gerekli algoritma sonuçları eksik.")
        except Exception as e: print(f"Pivot tablo hatası: {e}")


        # (Analiz 7 - Raporu Kaydetme)
        try:
            df_results.to_csv(report_file, index=False, float_format="%.0f")
            print(f"\n--- TÜM SONUÇLAR BAŞARIYLA '{report_file}' DOSYASINA KAYDEDİLDİ ---")
        except Exception as e: print(f"Rapor dosyası kaydedilemedi: {e}")

    except FileNotFoundError as e:
        print(f"\n!!! HATA: Gerekli dosya bulunamadı: {e}")
        print("Veritabanı dosyasının 'data/' klasöründe olduğundan emin olun.")
    except ImportError as e:
         print(f"HATA: Import hatası: {e}")
         print("Gerekli kütüphanelerin kurulu olduğundan emin olun (requirements.txt).")
    except Exception as e:
        print(f"\n!!! Simülasyon sırasında beklenmedik hata: {e}")
        import traceback
        traceback.print_exc() # Detaylı hata çıktısı için


if __name__ == "__main__":
    main_simulation()