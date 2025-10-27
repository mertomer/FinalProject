import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Türkçe karakter desteği için font ayarı
plt.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """CSV dosyasını yükler ve veriyi hazırlar."""
    csv_file = "simulasyon_sonuclari_detayli.csv"
    
    if not os.path.exists(csv_file):
        print(f"HATA: '{csv_file}' dosyası bulunamadı!")
        print("Önce simulate.py'yi çalıştırarak verileri oluşturun.")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Veri tiplerini düzelt (büyük sayılar için float kullan)
    df['n_blocks'] = pd.to_numeric(df['n_blocks'])
    df['toplam_odul'] = pd.to_numeric(df['toplam_odul'], errors='coerce')
    df['kapasite_doluluk_yuzdesi'] = pd.to_numeric(df['kapasite_doluluk_yuzdesi'])
    df['secilen_islem_sayisi'] = pd.to_numeric(df['secilen_islem_sayisi'])
    df['geride_kalan_islem_sayisi'] = pd.to_numeric(df['geride_kalan_islem_sayisi'])
    df['ortalama_odul_per_tx'] = pd.to_numeric(df['ortalama_odul_per_tx'], errors='coerce')
    df['ortalama_agirlik_per_tx'] = pd.to_numeric(df['ortalama_agirlik_per_tx'], errors='coerce')
    
    # Sütun adlarını standartlaştır (simulate.py'deki sütun adlarıyla uyumlu)
    df['islem_odulleri'] = df['toplam_odul']
    df['kullanilan_kapasite_yuzdesi'] = df['kapasite_doluluk_yuzdesi']
    df['bekleyen_islem_sayisi'] = df['geride_kalan_islem_sayisi']
    
    # NaN değerleri kontrol et
    if df['islem_odulleri'].isna().any():
        print("⚠️  Bazı ödül değerleri NaN olarak işaretlendi. Bu değerler atlanacak.")
        df = df.dropna(subset=['islem_odulleri'])
    
    print(f"✅ Veri yüklendi: {len(df)} satır")
    print(f"📊 Blok aralığı: {df['n_blocks'].min()}-{df['n_blocks'].max()}")
    print(f"🔧 Algoritma sayısı: {df['algoritma'].nunique()}")
    print(f"📋 Mevcut sütunlar: {list(df.columns)}")
    
    return df

def create_block_rewards_comparison(df):
    """Blok ödülleri karşılaştırma grafiği."""
    plt.figure(figsize=(14, 8))
    
    # Her algoritma için renk tanımla
    colors = {
        'Açgözlü (Greedy)': '#2E8B57',      # Deniz yeşili
        'Genetik Algoritma (YZ)': '#4169E1', # Kraliyet mavisi
        'Rastgele': '#DC143C',               # Kırmızı
        'Benzetilmiş Tavlama (YZ)': '#FF8C00' # Turuncu
    }
    
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        plt.plot(algo_data['n_blocks'], algo_data['islem_odulleri']/1e18, 
                marker='o', linewidth=2.5, markersize=6, 
                label=algo, color=colors.get(algo, '#000000'))
    
    plt.xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Toplam Blok Ödülü (×10¹⁸)', fontsize=12, fontweight='bold')
    plt.title('Blok Ödülleri Karşılaştırması\n(Algoritma Performansı)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Y eksenini daha okunabilir yap
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    plt.savefig('blok_odulleri_karsilastirma.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Blok ödülleri karşılaştırma grafiği oluşturuldu: blok_odulleri_karsilastirma.png")

def create_capacity_utilization_chart(df):
    """Kapasite kullanım yüzdesi grafiği."""
    plt.figure(figsize=(14, 8))
    
    # Pivot tablo oluştur
    pivot_df = df.pivot(index='n_blocks', columns='algoritma', values='kullanilan_kapasite_yuzdesi')
    
    # Heatmap oluştur
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                cbar_kws={'label': 'Kapasite Kullanım Yüzdesi (%)'})
    
    plt.title('Kapasite Kullanım Yüzdesi Heatmap\n(Blok Sayısı vs Algoritma)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Algoritma', fontsize=12, fontweight='bold')
    plt.ylabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('kapasite_kullanim_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Kapasite kullanım heatmap'i oluşturuldu: kapasite_kullanim_heatmap.png")

def create_transaction_waiting_analysis(df):
    """İşlem bekleme süreleri analizi."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = {
        'Açgözlü (Greedy)': '#2E8B57',
        'Genetik Algoritma (YZ)': '#4169E1',
        'Rastgele': '#DC143C',
        'Benzetilmiş Tavlama (YZ)': '#FF8C00'
    }
    
    # Seçilen işlem sayısı (İşlem işleme kapasitesi)
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax1.plot(algo_data['n_blocks'], algo_data['secilen_islem_sayisi'], 
                marker='o', linewidth=2.5, markersize=6, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax1.set_xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('İşlenen İşlem Sayısı', fontsize=12, fontweight='bold')
    ax1.set_title('İşlem İşleme Kapasitesi Karşılaştırması\n(Blok Başına İşlenen İşlemler)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bekleyen işlem sayısı (İşlem bekleme süresi temsili)
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax2.plot(algo_data['n_blocks'], algo_data['bekleyen_islem_sayisi'], 
                marker='s', linewidth=2.5, markersize=6, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax2.set_xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bekleyen İşlem Sayısı', fontsize=12, fontweight='bold')
    ax2.set_title('İşlem Bekleme Süreleri Analizi\n(Bekleyen İşlemler)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('islem_bekleme_analizi.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ İşlem bekleme süreleri analizi grafiği oluşturuldu: islem_bekleme_analizi.png")

def create_algorithm_efficiency_chart(df):
    """Algoritma verimliliği analizi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Toplam ödül karşılaştırması
    avg_performance = df.groupby('algoritma')['islem_odulleri'].mean().sort_values(ascending=True)
    bars1 = ax1.barh(range(len(avg_performance)), avg_performance/1e18, 
                    color=['#DC143C', '#FF8C00', '#4169E1', '#2E8B57'])
    
    for i, (algo, value) in enumerate(avg_performance.items()):
        ax1.text(value/1e18 + 0.1, i, f'{value/1e18:.1f}×10¹⁸', 
                va='center', fontweight='bold')
    
    ax1.set_yticks(range(len(avg_performance)))
    ax1.set_yticklabels(avg_performance.index)
    ax1.set_xlabel('Ortalama Toplam Ödül (×10¹⁸)', fontsize=12, fontweight='bold')
    ax1.set_title('Toplam Ödül Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Ortalama ödül/işlem karşılaştırması (Kalite metriği)
    avg_quality = df.groupby('algoritma')['ortalama_odul_per_tx'].mean().sort_values(ascending=True)
    bars2 = ax2.barh(range(len(avg_quality)), avg_quality/1e15, 
                    color=['#DC143C', '#FF8C00', '#4169E1', '#2E8B57'])
    
    for i, (algo, value) in enumerate(avg_quality.items()):
        ax2.text(value/1e15 + 0.1, i, f'{value/1e15:.1f}×10¹⁵', 
                va='center', fontweight='bold')
    
    ax2.set_yticks(range(len(avg_quality)))
    ax2.set_yticklabels(avg_quality.index)
    ax2.set_xlabel('Ortalama Ödül/İşlem (×10¹⁵)', fontsize=12, fontweight='bold')
    ax2.set_title('İşlem Kalitesi Karşılaştırması\n(Yüksek = Daha Kaliteli İşlemler)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Algoritma Verimliliği ve Kalite Analizi', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('algoritma_verimliligi.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Algoritma verimliliği grafiği oluşturuldu: algoritma_verimliligi.png")

def create_transaction_weight_analysis(df):
    """İşlem ağırlığı analizi - simulate.py'deki yeni metrik."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    colors = {
        'Açgözlü (Greedy)': '#2E8B57',
        'Genetik Algoritma (YZ)': '#4169E1',
        'Rastgele': '#DC143C',
        'Benzetilmiş Tavlama (YZ)': '#FF8C00'
    }
    
    # 1. Ortalama ağırlık/işlem karşılaştırması
    avg_weight_per_tx = df.groupby('algoritma')['ortalama_agirlik_per_tx'].mean().sort_values(ascending=True)
    bars1 = ax1.barh(range(len(avg_weight_per_tx)), avg_weight_per_tx, 
                    color=['#DC143C', '#FF8C00', '#4169E1', '#2E8B57'])
    
    for i, (algo, value) in enumerate(avg_weight_per_tx.items()):
        ax1.text(value + 1000, i, f'{value:,.0f}', 
                va='center', fontweight='bold')
    
    ax1.set_yticks(range(len(avg_weight_per_tx)))
    ax1.set_yticklabels(avg_weight_per_tx.index)
    ax1.set_xlabel('Ortalama Ağırlık/İşlem', fontsize=12, fontweight='bold')
    ax1.set_title('İşlem Ağırlığı Tercihi\n(Yüksek = Daha Ağır İşlemler Seçildi)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Blok sayısına göre ağırlık değişimi
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo].sort_values('n_blocks')
        ax2.plot(algo_data['n_blocks'], algo_data['ortalama_agirlik_per_tx'], 
                marker='o', linewidth=2.5, markersize=6, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax2.set_xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Ortalama Ağırlık/İşlem', fontsize=12, fontweight='bold')
    ax2.set_title('Blok Sayısına Göre İşlem Ağırlığı Değişimi', 
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('İşlem Ağırlığı Analizi (simulate.py Yeni Metrikleri)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('islem_agirlik_analizi.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ İşlem ağırlığı analizi grafiği oluşturuldu: islem_agirlik_analizi.png")

def create_block_size_scalability_analysis(df):
    """Blok sayısı ölçeklenebilirlik analizi."""
    plt.figure(figsize=(14, 8))
    
    # Her algoritma için blok sayısına göre performans artışını hesapla
    colors = {
        'Açgözlü (Greedy)': '#2E8B57',
        'Genetik Algoritma (YZ)': '#4169E1',
        'Rastgele': '#DC143C',
        'Benzetilmiş Tavlama (YZ)': '#FF8C00'
    }
    
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo].sort_values('n_blocks')
        
        # Performans artışını hesapla (5 bloktan başlayarak)
        base_reward = algo_data.iloc[0]['islem_odulleri']
        performance_growth = ((algo_data['islem_odulleri'] - base_reward) / base_reward) * 100
        
        plt.plot(algo_data['n_blocks'], performance_growth, 
                marker='o', linewidth=2.5, markersize=6, 
                label=algo, color=colors.get(algo, '#000000'))
    
    plt.xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Performans Artışı (%)', fontsize=12, fontweight='bold')
    plt.title('Blok Sayısı Ölçeklenebilirlik Analizi\n(5 Bloktan Başlayarak Performans Artışı)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Y eksenini daha okunabilir yap
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    plt.tight_layout()
    plt.savefig('blok_sayisi_olceklendirme.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Blok sayısı ölçeklenebilirlik analizi grafiği oluşturuldu: blok_sayisi_olceklendirme.png")

def create_comprehensive_simulation_dashboard(df):
    """Kapsamlı simülasyon dashboard'u."""
    fig = plt.figure(figsize=(24, 16))
    
    # 3x2 grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    colors = {'Açgözlü (Greedy)': '#2E8B57', 'Genetik Algoritma (YZ)': '#4169E1', 
              'Rastgele': '#DC143C', 'Benzetilmiş Tavlama (YZ)': '#FF8C00'}
    
    # 1. Blok Ödülleri Karşılaştırması
    ax1 = fig.add_subplot(gs[0, 0])
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax1.plot(algo_data['n_blocks'], algo_data['islem_odulleri']/1e18, 
                marker='o', linewidth=2, markersize=4, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax1.set_title('Blok Ödülleri Karşılaştırması', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Blok Sayısı (N)')
    ax1.set_ylabel('Toplam Ödül (×10¹⁸)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. İşlem İşleme Kapasitesi
    ax2 = fig.add_subplot(gs[0, 1])
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax2.plot(algo_data['n_blocks'], algo_data['secilen_islem_sayisi'], 
                marker='o', linewidth=2, markersize=4, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax2.set_title('İşlem İşleme Kapasitesi', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Blok Sayısı (N)')
    ax2.set_ylabel('İşlenen İşlem Sayısı')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. İşlem Bekleme Süreleri
    ax3 = fig.add_subplot(gs[1, 0])
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax3.plot(algo_data['n_blocks'], algo_data['bekleyen_islem_sayisi'], 
                marker='s', linewidth=2, markersize=4, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax3.set_title('İşlem Bekleme Süreleri', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Blok Sayısı (N)')
    ax3.set_ylabel('Bekleyen İşlem Sayısı')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Kapasite Kullanımı Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    pivot_df = df.pivot(index='n_blocks', columns='algoritma', values='kullanilan_kapasite_yuzdesi')
    sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax4, cbar=False)
    ax4.set_title('Kapasite Kullanımı (%)', fontweight='bold', fontsize=12)
    
    # 5. İşlem Kalitesi (Ortalama Ödül/İşlem)
    ax5 = fig.add_subplot(gs[2, 0])
    avg_quality = df.groupby('algoritma')['ortalama_odul_per_tx'].mean().sort_values(ascending=True)
    bars = ax5.barh(range(len(avg_quality)), avg_quality/1e15, 
                   color=['#DC143C', '#FF8C00', '#4169E1', '#2E8B57'])
    ax5.set_yticks(range(len(avg_quality)))
    ax5.set_yticklabels(avg_quality.index)
    ax5.set_title('İşlem Kalitesi (Ödül/İşlem)', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Ortalama Ödül/İşlem (×10¹⁵)')
    
    # 6. İşlem Ağırlığı Analizi
    ax6 = fig.add_subplot(gs[2, 1])
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo].sort_values('n_blocks')
        ax6.plot(algo_data['n_blocks'], algo_data['ortalama_agirlik_per_tx'], 
                marker='o', linewidth=2, markersize=4, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax6.set_title('İşlem Ağırlığı Değişimi', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Blok Sayısı (N)')
    ax6.set_ylabel('Ortalama Ağırlık/İşlem')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Kapsamlı Simülasyon Analizi Dashboard\n(Blok Sayısı, Blok Ödülleri, İşlem Bekleme Süreleri)', 
                 fontsize=16, fontweight='bold')
    plt.savefig('kapsamli_simulasyon_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Kapsamlı simülasyon dashboard'u oluşturuldu: kapsamli_simulasyon_dashboard.png")

def main():
    """Ana fonksiyon - tüm grafikleri oluşturur."""
    print("=" * 60)
    print("📊 SİMÜLASYON SONUÇLARI GÖRSEL ANALİZİ")
    print("=" * 60)
    
    # Veriyi yükle
    df = load_and_prepare_data()
    if df is None:
        return
    
    print("\n🎨 Simülasyon analiz grafikleri oluşturuluyor...")
    
    # 1. Blok ödülleri karşılaştırması
    create_block_rewards_comparison(df)
    
    # 2. İşlem bekleme süreleri analizi
    create_transaction_waiting_analysis(df)
    
    # 3. Kapasite kullanım heatmap
    create_capacity_utilization_chart(df)
    
    # 4. Algoritma verimliliği
    create_algorithm_efficiency_chart(df)
    
    # 5. İşlem ağırlığı analizi (simulate.py'deki yeni metrik)
    create_transaction_weight_analysis(df)
    
    # 6. Blok sayısı ölçeklenebilirlik analizi
    create_block_size_scalability_analysis(df)
    
    # 7. Kapsamlı simülasyon dashboard
    create_comprehensive_simulation_dashboard(df)
    
    print("\n" + "=" * 60)
    print("✅ TÜM SİMÜLASYON ANALİZ GRAFİKLERİ BAŞARIYLA OLUŞTURULDU!")
    print("=" * 60)
    print("📁 Oluşturulan dosyalar:")
    print("   • blok_odulleri_karsilastirma.png - Blok ödülleri karşılaştırması")
    print("   • islem_bekleme_analizi.png - İşlem bekleme süreleri analizi")
    print("   • kapasite_kullanim_heatmap.png - Kapasite kullanım heatmap'i")
    print("   • algoritma_verimliligi.png - Algoritma verimliliği ve kalite analizi")
    print("   • islem_agirlik_analizi.png - İşlem ağırlığı analizi (yeni metrik)")
    print("   • blok_sayisi_olceklendirme.png - Blok sayısı ölçeklenebilirlik")
    print("   • kapsamli_simulasyon_dashboard.png - Kapsamlı analiz dashboard'u")
    print("\n🎯 Bu grafikler simulate.py'deki tüm metrikleri kullanarak")
    print("   blok sayısı, blok ödülleri, işlem bekleme süreleri ve yeni kalite")
    print("   metriklerini kapsamlı olarak analiz etmenizi sağlar!")

if __name__ == "__main__":
    main()
