# scripts/visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# --- Ayarlar ---
# Proje kök dizinini bul (visualize_results.py'nin bir üst dizini)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
CSV_FILE = os.path.join(RESULTS_DIR, "simulasyon_sonuclari_detayli_tps.csv")

# Renk paleti
COLORS = {
    'Açgözlü (Greedy)': '#2E8B57',      # Deniz yeşili
    'Genetik Algoritma (YZ)': '#4169E1', # Kraliyet mavisi
    'Rastgele': '#DC143C',               # Kırmızı
    'Benzetilmiş Tavlama (YZ)': '#FF8C00', # Turuncu
    'RL PPO (YZ)': '#8A2BE2'            # Mor (RL için)
}

# --- Veri Yükleme ---
def load_and_prepare_data(csv_path=CSV_FILE):
    """CSV dosyasını yükler ve veriyi hazırlar."""
    if not os.path.exists(csv_path):
        print(f"HATA: '{csv_path}' dosyası bulunamadı!")
        print("Önce ana simulate.py'yi çalıştırarak verileri oluşturun.")
        return None

    df = pd.read_csv(csv_path)

    # Veri tiplerini düzelt
    numeric_cols = ['n_blocks', 'toplam_odul', 'kapasite_doluluk_yuzdesi',
                    'secilen_islem_sayisi', 'geride_kalan_islem_sayisi',
                    'ortalama_odul_per_tx', 'ortalama_agirlik_per_tx', 'tps']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sütun adlarını standartlaştır
    df['islem_odulleri'] = df['toplam_odul']
    df['kullanilan_kapasite_yuzdesi'] = df['kapasite_doluluk_yuzdesi']
    df['bekleyen_islem_sayisi'] = df['geride_kalan_islem_sayisi']
    df['ortalama_odul_bolu_islem'] = df['ortalama_odul_per_tx']
    df['ortalama_agirlik_bolu_islem'] = df['ortalama_agirlik_per_tx']

    # NaN kontrolü
    if df['islem_odulleri'].isna().any():
        print("⚠️ Bazı ödül değerleri NaN, bu satırlar atlanacak.")
        df = df.dropna(subset=['islem_odulleri'])

    print(f"✅ Veri yüklendi: {len(df)} satır")
    return df

# --- Grafik Fonksiyonları ---
# (Buraya önceki visualize_results.py'deki tüm create_... fonksiyonlarını
#  kopyalayıp yapıştırın. Sadece plt.savefig satırlarını güncelleyin.)

def save_plot(filename, results_dir=RESULTS_DIR):
    """Grafiği belirtilen dizine kaydeder."""
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close() # Grafiği kapatalım ki bellekte yer kaplamasın
        print(f"✅ Grafik kaydedildi: {filepath}")
    except Exception as e:
        print(f"HATA: Grafik kaydedilemedi ({filename}): {e}")

# Örnek Güncellenmiş Grafik Fonksiyonu:
def create_block_rewards_comparison(df):
    plt.figure(figsize=(12, 7)) # Boyut ayarlandı
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        plt.plot(algo_data['n_blocks'], algo_data['islem_odulleri']/1e18, # Ether'e çevir
                 marker='o', linewidth=2, markersize=5,
                 label=algo, color=COLORS.get(algo, '#000000'))
    plt.xlabel('Blok Sayısı (N)'); plt.ylabel('Toplam Blok Ödülü (ETH)')
    plt.title('Blok Ödülleri Karşılaştırması'); plt.legend()
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f} ETH'))
    save_plot('blok_odulleri_karsilastirma.png') # Kaydetme fonksiyonunu çağır

# --- DİĞER TÜM create_... GRAFİK FONKSİYONLARINI BURAYA EKLEYİN ---
# --- VE HER BİRİNİN SONUNDAKİ plt.savefig(...) SATIRINI ---
# --- save_plot('dosya_adi.png') ŞEKLİNDE GÜNCELLEYİN ---

def create_transaction_waiting_analysis(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax1.plot(algo_data['n_blocks'], algo_data['secilen_islem_sayisi'], marker='o', linewidth=2, markersize=5, label=algo, color=COLORS.get(algo, '#000000'))
    ax1.set_ylabel('İşlenen İşlem Sayısı'); ax1.set_title('İşlem İşleme Kapasitesi')
    ax1.legend(); ax1.grid(True, alpha=0.4)
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax2.plot(algo_data['n_blocks'], algo_data['bekleyen_islem_sayisi'], marker='s', linewidth=2, markersize=5, label=algo, color=COLORS.get(algo, '#000000'))
    ax2.set_xlabel('Blok Sayısı (N)'); ax2.set_ylabel('Bekleyen İşlem Sayısı')
    ax2.set_title('İşlem Bekleme Süreleri (Bekleyen İşlemler)')
    ax2.legend(); ax2.grid(True, alpha=0.4)
    plt.tight_layout()
    save_plot('islem_bekleme_analizi.png')

def create_capacity_utilization_chart(df):
    plt.figure(figsize=(10, 7))
    pivot_df = df.pivot(index='n_blocks', columns='algoritma', values='kullanilan_kapasite_yuzdesi')
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', cbar_kws={'label': '%'})
    plt.title('Kapasite Kullanım Yüzdesi (%)'); plt.xlabel('Algoritma'); plt.ylabel('Blok Sayısı (N)')
    plt.tight_layout()
    save_plot('kapasite_kullanim_heatmap.png')

def create_avg_reward_per_tx_plot(df):
    plt.figure(figsize=(12, 7))
    gwei_converter = 1e9
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        plt.plot(algo_data['n_blocks'], algo_data['ortalama_odul_bolu_islem'] / gwei_converter, marker='o', linewidth=2, markersize=5, label=algo, color=COLORS.get(algo, '#000000'))
    plt.xlabel('Blok Sayısı (N)'); plt.ylabel('Ortalama Ödül / İşlem (Gwei)')
    plt.title('İşlem Kalitesi (Ort. Ödül / İşlem)'); plt.legend()
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f} Gwei'))
    save_plot('ortalama_odul_per_tx.png')

def create_avg_weight_per_tx_plot(df):
    plt.figure(figsize=(12, 7))
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        plt.plot(algo_data['n_blocks'], algo_data['ortalama_agirlik_bolu_islem'], marker='o', linewidth=2, markersize=5, label=algo, color=COLORS.get(algo, '#000000'))
    plt.xlabel('Blok Sayısı (N)'); plt.ylabel('Ortalama Ağırlık / İşlem (Gas)')
    plt.title('İşlem Ağırlık Tercihi (Ort. Gas / İşlem)'); plt.legend()
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    save_plot('ortalama_agirlik_per_tx.png')

def create_algorithm_efficiency_chart(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) # Yan yana
    avg_performance = df.groupby('algoritma')['islem_odulleri'].mean().sort_values(ascending=False) # Büyükten küçüğe
    ax1.bar(avg_performance.index, avg_performance / 1e18, color=[COLORS.get(algo, '#000000') for algo in avg_performance.index])
    ax1.set_ylabel('Ortalama Toplam Ödül (ETH)'); ax1.set_title('Ortalama Ödül Verimliliği')
    ax1.grid(True, alpha=0.4, axis='y'); ax1.tick_params(axis='x', rotation=15)

    avg_quality = df.groupby('algoritma')['ortalama_odul_bolu_islem'].mean().sort_values(ascending=False) # Büyükten küçüğe
    ax2.bar(avg_quality.index, avg_quality / 1e9, color=[COLORS.get(algo, '#000000') for algo in avg_quality.index]) # Gwei'ye çevir
    ax2.set_ylabel('Ortalama Ödül/İşlem (Gwei)'); ax2.set_title('Ortalama İşlem Kalitesi')
    ax2.grid(True, alpha=0.4, axis='y'); ax2.tick_params(axis='x', rotation=15)

    plt.suptitle('Algoritma Verimlilik ve Kalite Özeti (Ortalama)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Başlık için yer aç
    save_plot('algoritma_verimlilik_ozet.png')

def create_block_size_scalability_analysis(df):
    plt.figure(figsize=(12, 7))
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo].sort_values('n_blocks')
        base_reward = algo_data.iloc[0]['islem_odulleri']
        performance_growth = ((algo_data['islem_odulleri'] - base_reward) / base_reward) * 100 if base_reward > 0 else pd.Series([0] * len(algo_data))
        plt.plot(algo_data['n_blocks'], performance_growth, marker='o', linewidth=2, markersize=5, label=algo, color=COLORS.get(algo, '#000000'))
    plt.xlabel('Blok Sayısı (N)'); plt.ylabel('Performans Artışı (%)')
    plt.title(f'Blok Sayısı Ölçeklenebilirlik ({df["n_blocks"].min()} Bloktan Başlayarak % Artış)')
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    plt.tight_layout()
    save_plot('blok_sayisi_olceklendirme.png')

def create_tps_over_blocks(df):
    plt.figure(figsize=(12, 7))
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo].sort_values('n_blocks')
        plt.plot(algo_data['n_blocks'], algo_data['tps'], marker='o', linewidth=2, markersize=5, label=algo, color=COLORS.get(algo, '#000000'))
    plt.xlabel('Blok Sayısı (N)'); plt.ylabel('TPS (işlem/sn)')
    plt.title('n_blocks’e Göre TPS'); plt.legend()
    plt.grid(True, alpha=0.4); plt.tight_layout()
    save_plot('tps_nblocks.png')

def create_avg_tps_bar(df):
    plt.figure(figsize=(10, 7))
    avg_tps = df.groupby('algoritma')['tps'].mean().sort_values(ascending=False)
    colors = [COLORS.get(algo, '#000000') for algo in avg_tps.index]
    plt.bar(avg_tps.index, avg_tps.values, color=colors)
    plt.ylabel('Ortalama TPS (işlem/sn)'); plt.title('Algoritmalara Göre Ortalama TPS')
    plt.grid(True, alpha=0.4, axis='y'); plt.xticks(rotation=15)
    plt.tight_layout()
    save_plot('tps_ortalama_algoritmalar.png')

# Kapsamlı Dashboard (İsteğe bağlı, 8 grafik olabilir)
def create_comprehensive_dashboard(df):
    print("\n[Dashboard] Kapsamlı dashboard oluşturuluyor...")
    fig = plt.figure(figsize=(20, 25)) # Boyutu artır (4x2 grid için)
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.25) # 4 satır

    # Grafikleri ilgili subplot'lara yerleştir
    # (Önceki dashboard kodunu buraya kopyalayıp, ax parametrelerini vererek
    # ve yeni grafikler için yer açarak güncelleyin.)
    # Örnek:
    # ax1 = fig.add_subplot(gs[0, 0])
    # ... (blok ödülleri kodu, sonunda ax=ax1 ekleyerek) ...

    # Şimdilik basitçe fonksiyonları çağıralım (ayrı dosyalar daha iyi)
    print("Dashboard fonksiyonu henüz tam olarak güncellenmedi, ayrı grafikler oluşturuldu.")
    # plt.suptitle('Kapsamlı Simülasyon Analizi Dashboard', fontsize=16, fontweight='bold')
    # save_plot('kapsamli_simulasyon_dashboard.png')


# --- ANA ÇALIŞTIRMA ---
def main():
    print("=" * 60 + "\n📊 SİMÜLASYON SONUÇLARI GÖRSEL ANALİZİ\n" + "=" * 60)
    df = load_and_prepare_data()
    if df is None: return

    print("\n🎨 Analiz grafikleri oluşturuluyor...")
    # Fonksiyonları sırayla çağır
    create_block_rewards_comparison(df)
    create_transaction_waiting_analysis(df)
    create_capacity_utilization_chart(df)
    create_avg_reward_per_tx_plot(df)       # Yeni
    create_avg_weight_per_tx_plot(df)       # Yeni
    create_algorithm_efficiency_chart(df)   # Güncellendi
    create_block_size_scalability_analysis(df)
    create_tps_over_blocks(df)              # Yeni: TPS çizgisi
    create_avg_tps_bar(df)                  # Yeni: Ortalama TPS bar
    # create_comprehensive_dashboard(df) # Opsiyonel

    print("\n" + "=" * 60 + "\n✅ TÜM GRAFİKLER BAŞARIYLA OLUŞTURULDU!\n" + "=" * 60)
    print(f"📁 Grafikler '{RESULTS_DIR}' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()