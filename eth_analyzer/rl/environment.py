# eth_analyzer/rl/environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

from ..simulation.core import get_simulation_pool # Havuz oluşturma fonksiyonumuzu import ediyoruz
from ..config import DEFAULT_START_BLOCK        # Varsayılan başlangıç bloğu

class MempoolEnv(gym.Env):
    """
    Ethereum Blok Doldurma Simülasyonu için Özel Gymnasium Ortamı.
    Ajan, kapasiteyi aşmadan toplam ödülü maksimize edecek şekilde
    havuzdan işlem seçmeyi öğrenir.
    """
    metadata = {'render_modes': []} # Render desteklemiyoruz

    def __init__(self, start_block=DEFAULT_START_BLOCK, n_blocks=10, k_top_actions=20,
                 pool_df=None, total_capacity=None):
        super().__init__()

        self.start_block = start_block
        self.n_blocks = n_blocks
        self.k_top_actions = k_top_actions # Karar vermek için en iyi kaç işleme bakılacak?

        # 1. Havuzu ve Kapasiteyi Yükle (Ortamın temel verisi)
        # Not: Gerçek eğitimde burası daha dinamik olabilir, şimdilik sabit
        self.full_pool_df = None
        self.total_capacity = None

        if pool_df is not None:
            if total_capacity is None:
                raise ValueError("pool_df ile birlikte total_capacity de sağlanmalıdır.")
            self._initialize_from_pool(pool_df, total_capacity)
        else:
            print(f"RL Ortamı başlatılıyor: {n_blocks} blokluk havuz yükleniyor...")
            try:
                fetched_pool_df, fetched_capacity = get_simulation_pool(self.start_block, self.n_blocks)
                self._initialize_from_pool(fetched_pool_df, fetched_capacity)
                print(f"Havuz yüklendi: {len(self.full_pool_df)} işlem, Kapasite: {self.total_capacity:,}")
            except FileNotFoundError:
                print("HATA: Veritabanı bulunamadı. Lütfen önce FetchData çalıştırın.")
                raise
            except ValueError as e:
                print(f"HATA: {e}")
                raise

        # 2. Aksiyon Alanı (Action Space)
        # Ajan, o anki mevcut işlemler arasından en yoğun K tanesinden birini seçebilir.
        # Ya da hiçbirini seçmez (örn: hiçbiri sığmıyorsa veya stratejik olarak).
        # Aksiyon 0: Hiçbir şey yapma / Atla
        # Aksiyon 1..K: En yoğun 1. .. K. işlemi seçmeyi dene
        self.action_space = spaces.Discrete(self.k_top_actions + 1)

        # 3. Gözlem Alanı (Observation Space)
        # Ajanın durumu nasıl gördüğü. Sabit boyutlu bir vektör olmalı.
        # Örnek: [normalize_edilmiş_kalan_kapasite, normalize_edilmiş_kalan_tx_sayısı,
        #         normalize_edilmiş_en_yoğun_tx_ağırlığı, normalize_edilmiş_en_yoğun_tx_değeri]
        # Değerler 0-1 arasında olacak şekilde normalize edilecek.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Başlangıç durumu için `reset` çağrısı (best practice)
        self.reset()

    def _initialize_from_pool(self, pool_df, total_capacity):
        """Havuz verisini hazırlar ve yoğunluk sütununu ekler."""
        if pool_df is None or pool_df.empty or total_capacity is None or total_capacity <= 0:
            raise ValueError("Havuz boş veya kapasite sıfır. Ortam başlatılamadı.")

        prepared_df = pool_df.copy()
        prepared_df = prepared_df.reset_index(drop=True)
        if 'density' not in prepared_df.columns:
            prepared_df['density'] = prepared_df['value'] / (prepared_df['weight'] + 1e-9)

        self.full_pool_df = prepared_df
        self.total_capacity = total_capacity

    def _get_observation(self):
        """Mevcut durumdan gözlem vektörünü oluşturur."""
        
        # Kalan işlemleri al
        remaining_tx = self.current_pool_df.drop(self.selected_tx_indices, errors='ignore')
        num_remaining = len(remaining_tx)

        # Gözlem değerlerini hesapla
        norm_remaining_capacity = self.remaining_capacity / self.total_capacity
        norm_num_remaining = num_remaining / len(self.full_pool_df) if len(self.full_pool_df) > 0 else 0
        
        top_dense_tx_weight_norm = 0.0
        top_dense_tx_value_norm = 0.0
        
        if not remaining_tx.empty:
            # En yoğun işlemi bul (normalize etmek için)
            top_tx = remaining_tx.loc[remaining_tx['density'].idxmax()]
            # Ağırlığı kapasiteye göre normalize et
            top_dense_tx_weight_norm = top_tx['weight'] / self.total_capacity if self.total_capacity > 0 else 0
            # Değeri (ödülü) potansiyel max ödüle göre normalize et (basit bir tahmin)
            # Ya da daha basitçe, tüm havuzdaki max değere göre normalize edelim
            max_value_in_pool = self.full_pool_df['value'].max()
            top_dense_tx_value_norm = top_tx['value'] / max_value_in_pool if max_value_in_pool > 0 else 0

        # Vektörü oluştur ve tipi float32 yap
        obs = np.array([
            norm_remaining_capacity,
            norm_num_remaining,
            np.clip(top_dense_tx_weight_norm, 0, 1), # Kırpma (clip) ile 0-1 arasında kalmasını garantile
            np.clip(top_dense_tx_value_norm, 0, 1)
        ], dtype=np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        """Yeni bir blok doldurma bölümü (episode) başlatır."""
        super().reset(seed=seed)

        # Ortamı başlangıç durumuna getir
        self.remaining_capacity = self.total_capacity
        self.current_pool_df = self.full_pool_df.copy() # Mevcut bölüm için havuz
        self.selected_tx_indices = set() # Bu bölümde seçilenlerin indeksleri
        self.current_total_reward = 0

        # Başlangıç gözlemini al
        observation = self._get_observation()
        info = {} # Ek bilgi (opsiyonel)

        # print("Ortam sıfırlandı.") # Debug için
        return observation, info

    def step(self, action):
        """Ajanın seçtiği aksiyonu uygular."""
        
        reward = 0
        terminated = False # Bölüm bitti mi?
        truncated = False # Zaman sınırı aşıldı mı? (Bizde yok)
        info = {}       # Ek bilgi

        # Kalan işlemleri al
        remaining_tx_df = self.current_pool_df.drop(self.selected_tx_indices, errors='ignore')
        
        # Seçilecek işlemi belirle
        target_tx_index = -1
        chosen_tx = None

        if action == 0 or remaining_tx_df.empty:
            # Aksiyon 0: Hiçbir şey yapma / Atla VEYA işlem kalmadı
            # Ödül 0. Durum değişmez. Belki küçük bir ceza eklenebilir.
            pass # Veya reward = -1 gibi küçük bir ceza
        else:
            # Aksiyon 1..K: En yoğun K işlemden birini seçmeyi dene
            # Kalan işlemleri yoğunluğa göre sırala
            top_k_dense = remaining_tx_df.nlargest(self.k_top_actions, 'density')
            
            if action <= len(top_k_dense):
                # Seçilen aksiyon geçerli bir işlem sırasına denk geliyor
                chosen_tx_series = top_k_dense.iloc[action - 1] # action 1 -> index 0
                target_tx_index = chosen_tx_series.name # DataFrame'deki orijinal indeksi
                chosen_tx = chosen_tx_series.to_dict()

        # Eğer geçerli bir işlem seçildiyse ve sığıyorsa
        if chosen_tx and target_tx_index != -1:
            if chosen_tx['weight'] <= self.remaining_capacity:
                # İşlem sığıyor -> Ekle
                self.remaining_capacity -= chosen_tx['weight']
                reward = chosen_tx['value'] # Ödül = işlemin değeri
                self.current_total_reward += reward
                self.selected_tx_indices.add(target_tx_index)
                # print(f"Aksiyon {action}: İşlem {target_tx_index} eklendi. Ödül: {reward}. Kalan Kapasite: {self.remaining_capacity}") # Debug
            else:
                # İşlem sığmıyor -> Negatif ödül (ceza) verebiliriz
                reward = -0.1 * chosen_tx['value'] # Örn: Değerinin %10'u kadar ceza (opsiyonel)
                # print(f"Aksiyon {action}: İşlem {target_tx_index} sığmadı.") # Debug
                
        # Bölüm bitiş kontrolü:
        # 1. Hiç işlem kalmadıysa
        # 2. Veya kalan hiçbir işlem kalan kapasiteye sığmıyorsa
        remaining_after_action = self.current_pool_df.drop(self.selected_tx_indices, errors='ignore')
        if remaining_after_action.empty:
            terminated = True
            # print("Bölüm bitti: İşlem kalmadı.") # Debug
        else:
             # Kalan en küçük işlemin ağırlığını bul
            min_weight_remaining = remaining_after_action['weight'].min()
            if min_weight_remaining > self.remaining_capacity:
                 terminated = True
                 # print("Bölüm bitti: Kalan hiçbir işlem sığmıyor.") # Debug

        # Yeni gözlemi al
        observation = self._get_observation()
        
        # Ek bilgi (opsiyonel, debug veya analiz için)
        info['current_total_reward'] = self.current_total_reward
        info['remaining_capacity'] = self.remaining_capacity
        info['selected_count'] = len(self.selected_tx_indices)

        return observation, reward, terminated, truncated, info
