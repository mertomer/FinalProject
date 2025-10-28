# simulate.py (Ana Çalıştırıcı - Kök Dizin)
import pandas as pd
import numpy as np
import sys
import os
import time

# Proje kök dizinini Python yoluna ekle (gerekirse)
project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(project_root, 'src'))  # Eğer paket yapın farklıysa açın

# eth_analyzer paketinden gerekli modülleri ve ayarları import et
try:
    from eth_analyzer.config import (
        DEFAULT_START_BLOCK, SIMULATION_MIN_WINDOW, SIMULATION_MAX_WINDOW,
        REPORT_CSV_FILE
    )
    from eth_analyzer.simulation.core import get_simulation_pool
    from eth_analyzer.simulation.algorithms import (
        solve_random, solve_greedy,
        solve_genetic_algorithm, solve_simulated_annealing,
        solve_custom_rl  # RL FONKSİYONU
    )
except ImportError as e:
    print(f"HATA: Gerekli modüller import edilemedi: {e}")
    print("Proje yapısını ve __init__.py dosyalarını kontrol edin.")
    sys.exit(1)


def main_simulation():
    """
    Ana simülasyonu çalıştırır, algoritmaları kıyaslar ve detaylı raporlar (TPS dahil).
    """
    print("=" * 60)
    print("ANA SİMÜLASYON BAŞLATILIYOR (TPS Metriği Dahil)")
    print("=" * 60)

    # İstersen config sabitlerini kullan (burada açıkça belirtiyoruz):
    START_BLOCK = DEFAULT_START_BLOCK if 'DEFAULT_START_BLOCK' in globals() else 20866020
    MIN_WINDOW = SIMULATION_MIN_WINDOW if 'SIMULATION_MIN_WINDOW' in globals() else 5
    MAX_WINDOW = SIMULATION_MAX_WINDOW if 'SIMULATION_MAX_WINDOW' in globals() else 20

    columns = [
        'n_blocks', 'algoritma', 'toplam_odul',
        'kapasite_doluluk_yuzdesi', 'secilen_islem_sayisi',
        'geride_kalan_islem_sayisi',
        'ortalama_odul_per_tx', 'ortalama_agirlik_per_tx',
        'tps',
        'kullanilan_kapasite', 'toplam_kapasite', 'havuzdaki_islem_sayisi'
    ]
    df_results = pd.DataFrame(columns=columns)

    try:
        for n_blocks in range(MIN_WINDOW, MAX_WINDOW + 1):
            print("\n" + "=" * 60 + f"\nPENCERE BOYUTU = {n_blocks}\n" + "=" * 60)

            print(f"\n[Core] Havuz oluşturuluyor...")
            pool_df, total_capacity = get_simulation_pool(START_BLOCK, n_blocks)
            if pool_df.empty or total_capacity == 0:
                continue
            havuzdaki_islem_sayisi = len(pool_df)

            print(f"\n[Algorithms] Algoritmalar çalıştırılıyor...")
            results_for_n = [
                solve_random(pool_df, total_capacity, n_blocks),
                solve_greedy(pool_df, total_capacity, n_blocks),
                solve_genetic_algorithm(pool_df, total_capacity, n_blocks),
                solve_simulated_annealing(pool_df, total_capacity, n_blocks),

                # --- RL PPO Model Karşılaştırması ---
                solve_custom_rl(
                    pool_df,
                    total_capacity,
                    n_blocks,
                    model_path="/Users/macbook/Desktop/Bitirme/FinalProject/ppo_mempool_v2.zip",
                    vecnorm_path="/Users/macbook/Desktop/Bitirme/FinalProject/vecnormalize_v2.pkl",
                ),
            ]

            print(f"\n--- N={n_blocks} İçin Sonuçlar Kaydediliyor ---")
            for res in results_for_n:
                res['n_blocks'] = n_blocks
                res['toplam_kapasite'] = total_capacity
                res['havuzdaki_islem_sayisi'] = havuzdaki_islem_sayisi
                res['geride_kalan_islem_sayisi'] = havuzdaki_islem_sayisi - res['secilen_islem_sayisi']
                df_results.loc[len(df_results)] = res

        print("\n" + "=" * 60 + "\nTÜM SİMÜLASYONLARIN ÖZET RAPORU (Task 4.2)\n" + "=" * 60)

        if df_results.empty:
            print("Hiçbir sonuç üretilmedi.")
            return

        for col in [
            'toplam_odul', 'n_blocks', 'kapasite_doluluk_yuzdesi', 'secilen_islem_sayisi',
            'geride_kalan_islem_sayisi', 'ortalama_odul_per_tx', 'ortalama_agirlik_per_tx',
            'tps', 'kullanilan_kapasite', 'toplam_kapasite', 'havuzdaki_islem_sayisi'
        ]:
            df_results[col] = pd.to_numeric(df_results[col])

        # 1. Genel Ortalama Ödül
        print("\n--- ANALİZ 1: ORTALAMA 'BLOK ÖDÜLLERİ' SIRALAMASI ---")
        avg_odul = df_results.groupby('algoritma')['toplam_odul'].mean().sort_values(ascending=False)
        print(avg_odul.to_string(float_format=lambda x: f"{x:.0f}"))

        # 2. Ortalama Bekleyen İşlem
        print("\n--- ANALİZ 2: ORTALAMA 'GERİDE KALAN İŞLEM SAYISI' SIRALAMASI ---")
        avg_bekleyen = df_results.groupby('algoritma')['geride_kalan_islem_sayisi'].mean().sort_values(ascending=True)
        print(avg_bekleyen.to_string(float_format=lambda x: f"{x:.0f}"))

        # 3. Ortalama Ödül / İşlem
        print("\n--- ANALİZ 3: ORTALAMA 'ÖDÜL / İŞLEM' SIRALAMASI (Kalite Metriği) ---")
        avg_odul_tx = df_results.groupby('algoritma')['ortalama_odul_per_tx'].mean().sort_values(ascending=False)
        print(avg_odul_tx.to_string(float_format=lambda x: f"{x:.0f}"))

        # 4. Ortalama Ağırlık / İşlem
        print("\n--- ANALİZ 4: ORTALAMA 'AĞIRLIK / İŞLEM' SIRALAMASI ---")
        avg_agirlik_tx = df_results.groupby('algoritma')['ortalama_agirlik_per_tx'].mean().sort_values(ascending=False)
        print(avg_agirlik_tx.to_string(float_format=lambda x: f"{x:.0f}"))

        # 5. Ortalama TPS
        print("\n--- ANALİZ 5: ORTALAMA 'TPS (SİMÜLE EDİLMİŞ)' SIRALAMASI ---")
        print("(Yüksek = Daha İyi Hacim)")
        avg_tps = df_results.groupby('algoritma')['tps'].mean().sort_values(ascending=False)
        print(avg_tps.to_string(float_format=lambda x: f"{x:.2f} TPS"))

        # 6 ve 7: (İsteğe bağlı ayrıntı analizleri — burada boş)
        print("\n--- ANALİZ 6 & 7: (Opsiyonel kıyaslamalar) ---")

        # 8. Raporu Kaydet
        try:
            rapor_dosyasi = "simulasyon_sonuclari_detayli_tps.csv"
            df_results = df_results[columns]
            df_results.to_csv(rapor_dosyasi, index=False, float_format="%.2f")
            print(f"\n--- TÜM SONUÇLAR BAŞARIYLA '{rapor_dosyasi}' DOSYASINA KAYDEDİLDİ ---")
        except Exception as e:
            print(f"Rapor dosyası kaydedilemedi: {e}")

    except FileNotFoundError:
        print("\n!!! HATA: 'ethereum_data.db' bulunamadı.")
    except Exception as e:
        print(f"\n!!! Simülasyon sırasında hata: {e}")


if __name__ == "__main__":
    main_simulation()
