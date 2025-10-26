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
    csv_file = "simulasyon_sonuclari.csv"
    
    if not os.path.exists(csv_file):
        print(f"HATA: '{csv_file}' dosyası bulunamadı!")
        print("Önce simulate.py'yi çalıştırarak verileri oluşturun.")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Veri tiplerini düzelt (büyük sayılar için float kullan)
    df['n_blocks'] = pd.to_numeric(df['n_blocks'])
    df['islem_odulleri'] = pd.to_numeric(df['islem_odulleri'], errors='coerce')
    df['kullanilan_kapasite_yuzdesi'] = pd.to_numeric(df['kullanilan_kapasite_yuzdesi'])
    df['secilen_islem_sayisi'] = pd.to_numeric(df['secilen_islem_sayisi'])
    df['bekleyen_islem_sayisi'] = pd.to_numeric(df['bekleyen_islem_sayisi'])
    
    # NaN değerleri kontrol et
    if df['islem_odulleri'].isna().any():
        print("⚠️  Bazı ödül değerleri NaN olarak işaretlendi. Bu değerler atlanacak.")
        df = df.dropna(subset=['islem_odulleri'])
    
    print(f"✅ Veri yüklendi: {len(df)} satır")
    print(f"📊 Blok aralığı: {df['n_blocks'].min()}-{df['n_blocks'].max()}")
    print(f"🔧 Algoritma sayısı: {df['algoritma'].nunique()}")
    
    return df

def create_performance_comparison_chart(df):
    """Algoritma performans karşılaştırma grafiği."""
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
    plt.ylabel('Toplam Ödül (×10¹⁸)', fontsize=12, fontweight='bold')
    plt.title('Algoritma Performans Karşılaştırması\n(Büyüyen Pencere Analizi)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Y eksenini daha okunabilir yap
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    plt.savefig('algoritma_performans_karsilastirma.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Performans karşılaştırma grafiği oluşturuldu: algoritma_performans_karsilastirma.png")

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

def create_transaction_selection_chart(df):
    """İşlem seçimi karşılaştırma grafiği."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = {
        'Açgözlü (Greedy)': '#2E8B57',
        'Genetik Algoritma (YZ)': '#4169E1',
        'Rastgele': '#DC143C',
        'Benzetilmiş Tavlama (YZ)': '#FF8C00'
    }
    
    # Seçilen işlem sayısı
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax1.plot(algo_data['n_blocks'], algo_data['secilen_islem_sayisi'], 
                marker='o', linewidth=2.5, markersize=6, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax1.set_xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Seçilen İşlem Sayısı', fontsize=12, fontweight='bold')
    ax1.set_title('Seçilen İşlem Sayısı Karşılaştırması', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Bekleyen işlem sayısı
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax2.plot(algo_data['n_blocks'], algo_data['bekleyen_islem_sayisi'], 
                marker='s', linewidth=2.5, markersize=6, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax2.set_xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bekleyen İşlem Sayısı', fontsize=12, fontweight='bold')
    ax2.set_title('Bekleyen İşlem Sayısı Karşılaştırması', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('islem_secimi_karsilastirma.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ İşlem seçimi karşılaştırma grafiği oluşturuldu: islem_secimi_karsilastirma.png")

def create_algorithm_efficiency_chart(df):
    """Algoritma verimliliği analizi."""
    plt.figure(figsize=(14, 8))
    
    # Her algoritma için ortalama performansı hesapla
    avg_performance = df.groupby('algoritma')['islem_odulleri'].mean().sort_values(ascending=True)
    
    # Bar chart oluştur
    bars = plt.barh(range(len(avg_performance)), avg_performance/1e18, 
                   color=['#DC143C', '#FF8C00', '#4169E1', '#2E8B57'])
    
    # Değerleri bar'ların üzerine yaz
    for i, (algo, value) in enumerate(avg_performance.items()):
        plt.text(value/1e18 + 0.1, i, f'{value/1e18:.1f}×10¹⁸', 
                va='center', fontweight='bold')
    
    plt.yticks(range(len(avg_performance)), avg_performance.index)
    plt.xlabel('Ortalama Toplam Ödül (×10¹⁸)', fontsize=12, fontweight='bold')
    plt.title('Algoritma Verimliliği Karşılaştırması\n(Ortalama Performans)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('algoritma_verimliligi.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Algoritma verimliliği grafiği oluşturuldu: algoritma_verimliligi.png")

def create_window_size_impact_chart(df):
    """Pencere boyutunun etkisi analizi."""
    plt.figure(figsize=(14, 8))
    
    # GA vs Greedy performans farkını hesapla
    greedy_data = df[df['algoritma'] == 'Açgözlü (Greedy)'].set_index('n_blocks')['islem_odulleri']
    ga_data = df[df['algoritma'] == 'Genetik Algoritma (YZ)'].set_index('n_blocks')['islem_odulleri']
    
    # Yüzdesel farkı hesapla
    performance_diff = ((ga_data - greedy_data) / greedy_data) * 100
    
    # Grafik oluştur
    plt.plot(performance_diff.index, performance_diff.values, 
             marker='o', linewidth=3, markersize=8, color='#4169E1')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Blok Sayısı (N)', fontsize=12, fontweight='bold')
    plt.ylabel('Performans Farkı (%)', fontsize=12, fontweight='bold')
    plt.title('Pencere Boyutunun Genetik Algoritma Performansına Etkisi\n(GA vs Açgözlü)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Y eksenini daha okunabilir yap
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    plt.tight_layout()
    plt.savefig('pencere_boyutu_etkisi.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Pencere boyutu etkisi grafiği oluşturuldu: pencere_boyutu_etkisi.png")

def create_summary_dashboard(df):
    """Özet dashboard oluştur."""
    fig = plt.figure(figsize=(20, 12))
    
    # 2x2 grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Performans karşılaştırması
    ax1 = fig.add_subplot(gs[0, 0])
    colors = {'Açgözlü (Greedy)': '#2E8B57', 'Genetik Algoritma (YZ)': '#4169E1', 
              'Rastgele': '#DC143C', 'Benzetilmiş Tavlama (YZ)': '#FF8C00'}
    
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax1.plot(algo_data['n_blocks'], algo_data['islem_odulleri']/1e18, 
                marker='o', linewidth=2, markersize=4, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax1.set_title('Performans Karşılaştırması', fontweight='bold')
    ax1.set_xlabel('Blok Sayısı')
    ax1.set_ylabel('Ödül (×10¹⁸)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Kapasite kullanımı
    ax2 = fig.add_subplot(gs[0, 1])
    pivot_df = df.pivot(index='n_blocks', columns='algoritma', values='kullanilan_kapasite_yuzdesi')
    sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax2, cbar=False)
    ax2.set_title('Kapasite Kullanımı (%)', fontweight='bold')
    
    # 3. İşlem seçimi
    ax3 = fig.add_subplot(gs[1, 0])
    for algo in df['algoritma'].unique():
        algo_data = df[df['algoritma'] == algo]
        ax3.plot(algo_data['n_blocks'], algo_data['secilen_islem_sayisi'], 
                marker='o', linewidth=2, markersize=4, 
                label=algo, color=colors.get(algo, '#000000'))
    
    ax3.set_title('Seçilen İşlem Sayısı', fontweight='bold')
    ax3.set_xlabel('Blok Sayısı')
    ax3.set_ylabel('İşlem Sayısı')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Verimlilik
    ax4 = fig.add_subplot(gs[1, 1])
    avg_performance = df.groupby('algoritma')['islem_odulleri'].mean().sort_values(ascending=True)
    bars = ax4.barh(range(len(avg_performance)), avg_performance/1e18, 
                   color=['#DC143C', '#FF8C00', '#4169E1', '#2E8B57'])
    ax4.set_yticks(range(len(avg_performance)))
    ax4.set_yticklabels(avg_performance.index)
    ax4.set_title('Ortalama Verimlilik', fontweight='bold')
    ax4.set_xlabel('Ödül (×10¹⁸)')
    
    plt.suptitle('Simülasyon Sonuçları Özet Dashboard', fontsize=16, fontweight='bold')
    plt.savefig('simulasyon_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Özet dashboard oluşturuldu: simulasyon_dashboard.png")

def main():
    """Ana fonksiyon - tüm grafikleri oluşturur."""
    print("=" * 60)
    print("📊 SİMÜLASYON SONUÇLARI GÖRSEL ANALİZİ")
    print("=" * 60)
    
    # Veriyi yükle
    df = load_and_prepare_data()
    if df is None:
        return
    
    print("\n🎨 Grafikler oluşturuluyor...")
    
    # 1. Performans karşılaştırması
    create_performance_comparison_chart(df)
    
    # 2. Kapasite kullanım heatmap
    create_capacity_utilization_chart(df)
    
    # 3. İşlem seçimi karşılaştırması
    create_transaction_selection_chart(df)
    
    # 4. Algoritma verimliliği
    create_algorithm_efficiency_chart(df)
    
    # 5. Pencere boyutu etkisi
    create_window_size_impact_chart(df)
    
    # 6. Özet dashboard
    create_summary_dashboard(df)
    
    print("\n" + "=" * 60)
    print("✅ TÜM GRAFİKLER BAŞARIYLA OLUŞTURULDU!")
    print("=" * 60)
    print("📁 Oluşturulan dosyalar:")
    print("   • algoritma_performans_karsilastirma.png")
    print("   • kapasite_kullanim_heatmap.png")
    print("   • islem_secimi_karsilastirma.png")
    print("   • algoritma_verimliligi.png")
    print("   • pencere_boyutu_etkisi.png")
    print("   • simulasyon_dashboard.png")
    print("\n🎯 Bu grafikler simülasyon sonuçlarınızı görsel olarak analiz etmenizi sağlar!")

if __name__ == "__main__":
    main()
