# Ethereum Blok Veri Çekme Projesi (Cloudflare RPC)

Bu proje, belirli bir tarihteki Ethereum bloklarını ve bu bloklara ait tüm işlem (transaction) detaylarını Cloudflare'in ücretsiz RPC düğümünü kullanarak çeker ve bir SQLite veritabanına kaydeder.

## Teknoloji Stack'i
* **Dil:** Python 3.x
* **Kütüphaneler:** `requests`
* **Veri Kaynağı:** Cloudflare Ethereum RPC (`https://cloudflare-eth.com`)
* **Yardımcı API:** Etherscan (Sadece tarih -> blok no dönüşümü için)
* **Veritabanı:** SQLite (`ethereum_data.db` adında tek bir dosyada saklanır)

## Kurulum

1.  Proje dosyalarını içeren bu klasörde bir terminal açın.
2.  (Tavsiye Edilir) Bir sanal ortam (virtual environment) oluşturun:
    ```bash
    python -m venv venv
    ```
3.  Sanal ortamı aktive edin:
    * Windows: `venv\Scripts\activate`
    * macOS/Linux: `source venv/bin/activate`
    
4.  Gerekli kütüphaneleri `requirements.txt` dosyasından yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

## Kullanım (Adım Adım)

### Adım 1: Etherscan API Anahtarı ve Tarih Ayarı

**ÇOK ÖNEMLİ GÜVENLİK UYARISI:**
Daha önceki konuşmamızda API anahtarınızı (API Key) paylaştınız. Bu anahtar artık güvenli değildir. Lütfen **hemen** Etherscan hesabınıza gidin, o anahtarı silin (revoke) ve **yeni bir API anahtarı oluşturun.**

1.  `find_blocks.py` dosyasını açın.
2.  `YENI_API_KEYINIZI_BURAYA_GIRIN` yazan yere yeni Etherscan API anahtarınızı yapıştırın.
3.  `TARGET_YEAR`, `TARGET_MONTH`, `TARGET_DAY` değişkenlerini veri çekmek istediğiniz tarihle güncelleyin.

### Adım 2: Başlangıç/Bitiş Blok Numaralarını Bulma

1.  Terminalde `find_blocks.py` script'ini çalıştırın:
    ```bash
    python find_blocks.py
    ```
2.  Script size şuna benzer bir çıktı verecektir:
    ```
    ✅ BAŞLANGIÇ BLOĞU (START_BLOCK): 19998542
    ✅ BİTİŞ BLOĞU (END_BLOCK):    20005741
    ```
3.  Bu iki (START ve END) blok numarasını kopyalayın.

### Adım 3: Ana Veri Çekme Script'ini Ayarlama

1.  `fetch_data.py` dosyasını açın.
2.  Dosyanın en üstündeki `START_BLOCK = 0` ve `END_BLOCK = 0` değişkenlerini, Adım 2'de kopyaladığınız numaralarla değiştirin.

### Adım 4: Projeyi Çalıştırma

1.  Tüm dosyaları kaydettikten sonra, terminalde ana script'i çalıştırın:
    ```bash
    python fetch_data.py
    ```
2.  Script çalışmaya başlayacak ve `ethereum_data.db` adında bir veritabanı dosyası oluşturacaktır.

### ÖNEMLİ BEKLENTİLER:
* **BU İŞLEM ÇOK UZUN SÜRECEKTİR.** Bir günlük veri (yaklaşık 7200 blok) çekmek, Cloudflare'in hız limitlerine (rate limit) takılacağı için **birkaç saat** sürebilir.
* Konsolda **"Rate Limit (429) alındı... bekleniyor..."** mesajları göreceksiniz. **Bu bir HATA değildir.** Bu, script'in çalıştığını ve Cloudflare'in "yavaşla" komutuna uyduğunu gösterir.
* İşlem bittiğinde, tüm veriler `ethereum_data.db` dosyasında olacaktır. Bu dosyayı [DB Browser for SQLite](https://sqlitebrowser.org/) gibi ücretsiz bir araçla açıp inceleyebilirsiniz.