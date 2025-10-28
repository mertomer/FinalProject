# eth_analyzer/config_template.py
import os

# !!! BU DOSYAYI config.py OLARAK KOPYALAYİP API ANAHTARLARINIZI GİRİN !!!
ETHERSCAN_API_KEY = "YOUR_ETHERSCAN_API_KEY_HERE"
ALCHEMY_RPC_URL = "YOUR_ALCHEMY_RPC_URL_HERE"
ETHERSCAN_BASE_URL = "https://api.etherscan.io/v2/api"

# --- Veritabanı Ayarları ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_FILE_PATH = os.path.join(_PROJECT_ROOT, "data", "ethereum_data.db")

# --- Simülasyon Ayarları ---
DEFAULT_START_BLOCK = 20866020
SIMULATION_MIN_WINDOW = 5
SIMULATION_MAX_WINDOW = 20

# --- Varsayılan Tarih (Blok Bulucu İçin) ---
DEFAULT_TARGET_YEAR = 2024
DEFAULT_TARGET_MONTH = 10
DEFAULT_TARGET_DAY = 1

# --- API İstek Başlıkları ---
REQUEST_HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

# --- GA Parametreleri ---
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 50
GA_CROSSOVER_PROB = 0.7
GA_MUTATION_PROB = 0.1

# --- SA Parametreleri ---
SA_INITIAL_TEMP = 100.0
SA_MIN_TEMP = 1e-3
SA_ALPHA = 0.99

# --- Sonuç Dosyaları Yolu ---
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
REPORT_CSV_FILE = os.path.join(RESULTS_DIR, "simulasyon_sonuclari_detayli.csv")

# --- RL Modeli Dosya Yolları ---
RL_MODEL_PATH = os.path.join(RESULTS_DIR, "rl_mempool_ppo_50000steps.zip")
RL_VECNORMALIZE_PATH = os.path.join(RESULTS_DIR, "rl_mempool_ppo_50000steps_vecnormalize.pkl")
