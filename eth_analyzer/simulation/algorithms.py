import random
import math
import time
import pandas as pd
import numpy as np
import cloudpickle  # VecNormalize istatistiklerini okumak için

DEAP_AVAILABLE = True
try:
    from deap import base, creator, tools, algorithms as deap_alg
except ImportError:
    DEAP_AVAILABLE = False
    print("UYARI: 'deap' kütüphanesi bulunamadı. GA için yerel (DEAP'siz) çözüm çalışacak.")

try:
    from ..config import (
        GA_POPULATION_SIZE, GA_GENERATIONS, GA_CROSSOVER_PROB, GA_MUTATION_PROB,
        SA_INITIAL_TEMP, SA_MIN_TEMP, SA_ALPHA
    )
except ImportError:
    print("Uyarı: config.py bulunamadı veya GA/SA parametreleri eksik. Varsayılan değerler kullanılacak.")
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 50
    GA_CROSSOVER_PROB = 0.7
    GA_MUTATION_PROB = 0.1
    SA_INITIAL_TEMP = 100.0
    SA_MIN_TEMP = 1e-3
    SA_ALPHA = 0.99


# --- YARDIMCI FONKSİYONLAR ---
def calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity, n_blocks):
    """
    Ortak metrikleri (TPS dahil) hesaplar.
    """
    avg_reward_per_tx = total_value / selected_tx_count if selected_tx_count > 0 else 0.0
    avg_weight_per_tx = total_weight / selected_tx_count if selected_tx_count > 0 else 0.0
    cap_util_percent = (total_weight / capacity) * 100 if capacity > 0 else 0.0

    # TPS Hesaplaması (Ortalama blok süresi ~12 saniye)
    estimated_duration_seconds = n_blocks * 12
    tps = selected_tx_count / estimated_duration_seconds if estimated_duration_seconds > 0 else 0.0

    return avg_reward_per_tx, avg_weight_per_tx, cap_util_percent, tps


# ===== 1) Rastgele =====
def solve_random(transactions_df, capacity, n_blocks):
    if transactions_df.empty or capacity <= 0:
        return {
            "algoritma": "Rastgele", "toplam_odul": 0.0, "kullanilan_kapasite": 0.0,
            "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0,
            "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0,
            "tps": 0.0
        }

    shuffled_df = transactions_df.sample(frac=1)
    total_value, total_weight, selected_tx_count = 0.0, 0.0, 0

    for _, tx in shuffled_df.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1

    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(
        total_value, total_weight, selected_tx_count, capacity, n_blocks
    )

    return {
        "algoritma": "Rastgele", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps
    }


# ===== 2) Greedy =====
def solve_greedy(transactions_df, capacity, n_blocks):
    if transactions_df.empty or capacity <= 0:
        return {
            "algoritma": "Açgözlü (Greedy)", "toplam_odul": 0.0, "kullanilan_kapasite": 0.0,
            "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0,
            "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0,
            "tps": 0.0
        }

    df = transactions_df.copy()
    df['density'] = df['value'] / (df['weight'] + 1e-9)
    df_sorted = df.sort_values(by='density', ascending=False)

    total_value, total_weight, selected_tx_count = 0.0, 0.0, 0
    for _, tx in df_sorted.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1

    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(
        total_value, total_weight, selected_tx_count, capacity, n_blocks
    )

    return {
        "algoritma": "Açgözlü (Greedy)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps
    }


# ===== 3) Genetik Algoritma (DEAP varsa/yoksa) =====
def solve_genetic_algorithm(transactions_df, capacity, n_blocks):
    if transactions_df.empty or capacity <= 0:
        return {
            "algoritma": "Genetik Algoritma (YZ)", "toplam_odul": 0.0, "kullanilan_kapasite": 0.0,
            "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0,
            "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0,
            "tps": 0.0
        }

    # --- Yerel GA (DEAP yoksa) ---
    if not DEAP_AVAILABLE:
        print("\n[GA] Yerel Genetik Algoritma (DEAP yok) çalıştırılıyor...")
        items = transactions_df[['weight', 'value']].to_dict('records')
        num_items = len(items)

        def evaluate(genome):
            w, v = 0, 0
            for i, bit in enumerate(genome):
                if bit == 1:
                    w += items[i]['weight']
                    v += items[i]['value']
            return (v if w <= capacity else 0), w

        def tournament_select(pop, k=3):
            selected = []
            for _ in range(len(pop)):
                contenders = random.sample(pop, min(k, len(pop)))
                best = max(contenders, key=lambda ind: ind['fitness'])
                selected.append({'genome': best['genome'][:], 'fitness': best['fitness'], 'weight': best['weight']})
            return selected

        def crossover(g1, g2):
            if random.random() < GA_CROSSOVER_PROB and len(g1) > 1:
                cx1 = random.randint(1, len(g1) - 1)
                cx2 = random.randint(cx1, len(g1) - 1)
                c1 = g1[:cx1] + g2[cx1:cx2] + g1[cx2:]
                c2 = g2[:cx1] + g1[cx1:cx2] + g2[cx2:]
                return c1, c2
            return g1[:], g2[:]

        def mutate(g):
            for i in range(len(g)):
                if random.random() < GA_MUTATION_PROB:
                    g[i] = 1 - g[i]
            return g

        population = []
        for _ in range(GA_POPULATION_SIZE):
            genome = [random.randint(0, 1) for _ in range(num_items)]
            fit, w = evaluate(genome)
            population.append({'genome': genome, 'fitness': fit, 'weight': w})

        best = max(population, key=lambda ind: ind['fitness'])
        start_time = time.time()
        for _ in range(GA_GENERATIONS):
            mating_pool = tournament_select(population)
            offspring = []
            for i in range(0, len(mating_pool), 2):
                p1 = mating_pool[i]['genome']
                p2 = mating_pool[i + 1]['genome'] if i + 1 < len(mating_pool) else mating_pool[0]['genome']
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1); c2 = mutate(c2)
                f1, w1 = evaluate(c1); f2, w2 = evaluate(c2)
                offspring.append({'genome': c1, 'fitness': f1, 'weight': w1})
                offspring.append({'genome': c2, 'fitness': f2, 'weight': w2})
            population = sorted(population + offspring, key=lambda ind: ind['fitness'], reverse=True)[:GA_POPULATION_SIZE]
            if population[0]['fitness'] > best['fitness']:
                best = population[0]
        print(f"[GA] Yerel GA {time.time() - start_time:.2f} saniyede tamamlandı.")

        best_genome = best['genome']
        total_value = best['fitness']
        total_weight = 0.0
        selected_tx_count = 0
        for i, bit in enumerate(best_genome):
            if bit == 1:
                total_weight += items[i]['weight']
                selected_tx_count += 1

        avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(
            total_value, total_weight, selected_tx_count, capacity, n_blocks
        )
        return {
            "algoritma": "Genetik Algoritma (YZ)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
            "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
            "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
            "tps": tps
        }

    # --- DEAP'li GA ---
    print("\n[GA] Genetik Algoritma (YZ) çalıştırılıyor...")
    items = transactions_df[['weight', 'value']].to_dict('records')
    num_items = len(items)

    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_items)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalKnapsack(individual):
        w, v = 0, 0
        for i in range(num_items):
            if individual[i] == 1:
                w += items[i]['weight']; v += items[i]['value']
        return (v,) if w <= capacity else (0,)

    toolbox.register("evaluate", evalKnapsack)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=GA_MUTATION_PROB)
    toolbox.register("select", tools.selTournament, tournsize=3)

    start_time = time.time()
    pop = toolbox.population(n=GA_POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x)/len(x) if x else 0)
    stats.register("max", lambda x: max(x) if x else 0)

    deap_alg.eaSimple(
        pop, toolbox,
        cxpb=GA_CROSSOVER_PROB,
        mutpb=GA_MUTATION_PROB,
        ngen=GA_GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=False
    )
    print(f"[GA] Genetik Algoritma {time.time() - start_time:.2f} saniyede tamamlandı.")

    best_individual = hof[0]
    total_value, total_weight, selected_tx_count = 0.0, 0.0, 0
    for i in range(num_items):
        if best_individual[i] == 1:
            total_weight += items[i]['weight']; total_value += items[i]['value']; selected_tx_count += 1

    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(
        total_value, total_weight, selected_tx_count, capacity, n_blocks
    )

    return {
        "algoritma": "Genetik Algoritma (YZ)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps
    }


# ===== 4) Simulated Annealing =====
def solve_simulated_annealing(transactions_df, capacity, n_blocks):
    if transactions_df.empty or capacity <= 0:
        return {
            "algoritma": "Benzetilmiş Tavlama (YZ)", "toplam_odul": 0.0, "kullanilan_kapasite": 0.0,
            "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0,
            "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0,
            "tps": 0.0
        }

    print("\n[SA] Benzetilmiş Tavlama (YZ) çalıştırılıyor...")
    items = transactions_df[['weight', 'value']].to_dict('records')
    num_items = len(items)

    def get_solution_stats(solution_indices):
        w, v = 0.0, 0.0
        for i in solution_indices:
            w += items[i]['weight']; v += items[i]['value']
        return w, v

    def create_initial_solution():
        sol, w = set(), 0.0
        indices = list(range(num_items)); random.shuffle(indices)
        for i in indices:
            if w + items[i]['weight'] <= capacity:
                w += items[i]['weight']; sol.add(i)
        _, v = get_solution_stats(sol)
        return sol, w, v

    T = SA_INITIAL_TEMP
    current_solution, current_weight, current_value = create_initial_solution()
    best_solution, best_value, best_weight = current_solution, current_value, current_weight

    start_time = time.time()
    max_iterations = 20000
    iter_count = 0

    while T > SA_MIN_TEMP and iter_count < max_iterations:
        new_solution = set(current_solution)

        p_remove = 0.5 if len(new_solution) > 0 else 0.0
        p_add = 0.5 if len(new_solution) < num_items else 1.0
        if len(new_solution) == 0:
            p_add = 1.0

        action = random.uniform(0, 1)
        if action < p_remove and len(new_solution) > 0:
            item_to_remove = random.choice(list(new_solution))
            new_solution.remove(item_to_remove)
        else:
            available_to_add = list(set(range(num_items)) - new_solution)
            if available_to_add:
                item_to_add = random.choice(available_to_add)
                new_solution.add(item_to_add)

        new_weight, new_value = get_solution_stats(new_solution)

        if new_weight <= capacity:
            delta_value = new_value - current_value
            if delta_value > 0 or (T > 1e-9 and random.random() < math.exp(delta_value / max(T, 1e-9))):
                current_solution, current_weight, current_value = new_solution, new_weight, new_value
                if current_value > best_value:
                    best_solution, best_weight, best_value = current_solution, new_weight, new_value

        T *= SA_ALPHA
        iter_count += 1

    print(f"[SA] Benzetilmiş Tavlama {time.time() - start_time:.2f} saniyede tamamlandı ({iter_count} iterasyon).")
    selected_tx_count = len(best_solution)

    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(
        best_value, best_weight, selected_tx_count, capacity, n_blocks
    )

    return {
        "algoritma": "Benzetilmiş Tavlama (YZ)", "toplam_odul": best_value, "kullanilan_kapasite": best_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps
    }


# ===== 5) RL (PPO) — Eğitim şemasıyla birebir inference =====
K_TOP_ACTIONS = 10     # Eğitimde: 0=pas + 10 aksiyon
PAS_ASSIST = True      # model 0 (pas) dese bile top-1 sığıyorsa al

def _load_vecnorm_stats(vecnorm_path):
    """
    VecNormalize .pkl içindeki gözlem RMS istatistiklerini (mean/var) güvenli okuyucu.
    """
    if not vecnorm_path:
        return None, None
    try:
        with open(vecnorm_path, "rb") as f:
            data = cloudpickle.load(f)
        rms = None
        if isinstance(data, dict) and "ob_rms" in data:
            rms = data["ob_rms"]
        elif isinstance(data, dict) and "obs_rms" in data:
            rms = data["obs_rms"]
        if rms is not None and hasattr(rms, "mean") and hasattr(rms, "var"):
            mean = np.array(rms.mean).astype(np.float32).flatten()
            var  = np.array(rms.var ).astype(np.float32).flatten()
            return mean, var
    except Exception as e:
        print(f"[RL] VecNormalize istatistikleri okunamadı: {e}")
    return None, None


class RLPolicyRunner:
    """
    PPO/VecNormalize ile eğitilmiş politikanın inference adaptörü.
    Eğitimdeki Gym ortamına sadık: Discrete(11) eylem (0=pas, 1..10 = top-k index)
    Gözlem: 4-boyut (norm_remaining_capacity, norm_num_remaining, top_w_norm, top_v_norm)
    """
    def __init__(self, model_path: str, vecnorm_path: str | None = None):
        self.model = None
        self.obs_mean, self.obs_var = _load_vecnorm_stats(vecnorm_path)
        self.expected_obs_shape = (4,)

        # SB3 2.3.0 geriye dönük yükleme uyumluluğu
        try:
            from stable_baselines3 import PPO
        except Exception as e:
            raise RuntimeError(f"Stable-Baselines3 yüklenemedi: {e}")

        custom_objects = {
            "lr_schedule": lambda _: 3e-4,
            "clip_range": 0.2,
            "clip_range_vf": None,
        }
        self.model = PPO.load(model_path, device="cpu", custom_objects=custom_objects)

    @staticmethod
    def make_obs(remaining_capacity: float, total_capacity: float,
                 remaining_df: pd.DataFrame, full_df: pd.DataFrame) -> np.ndarray:
        if total_capacity <= 0 or full_df.empty:
            return np.zeros(4, dtype=np.float32)

        norm_rem = remaining_capacity / (total_capacity + 1e-9)
        norm_num = len(remaining_df) / len(full_df)

        top_w_norm = 0.0
        top_v_norm = 0.0
        if not remaining_df.empty:
            top_tx = remaining_df.loc[remaining_df['density'].idxmax()]
            top_w_norm = float(top_tx['weight']) / (total_capacity + 1e-9)
            max_v = float(full_df['value'].max())
            top_v_norm = (float(top_tx['value']) / (max_v + 1e-9)) if max_v > 0 else 0.0

        obs = np.array([norm_rem, norm_num, top_w_norm, top_v_norm], dtype=np.float32)
        return obs

    def _maybe_norm(self, obs_1d: np.ndarray) -> np.ndarray:
        if self.obs_mean is None or self.obs_var is None:
            return obs_1d
        eps = 1e-8
        m = self.obs_mean[: len(obs_1d)]
        v = self.obs_var [: len(obs_1d)]
        return (obs_1d - m) / np.sqrt(v + eps)

    def predict_action(self, obs_1d: np.ndarray, valid_actions: int) -> int:
        x = obs_1d.reshape(1, -1)  # (n_env=1, obs_dim)
        action, _ = self.model.predict(x, deterministic=True)
        a = int(action[0])
        if a >= valid_actions:
            a = valid_actions - 1
        if a < 0:
            a = 0
        return a


def solve_custom_rl(pool_df: pd.DataFrame, capacity: int, n_blocks: int,
                    model_path: str, vecnorm_path: str | None = None):
    """
    Eğitim şemasına sadık RL seçim döngüsü:
    - Her adımda remaining_df -> density’e göre top-k
    - Obs(4) üret, (varsa) VecNormalize istatistikleriyle normalize et
    - Model action: 0=pas, 1..k = top-k index, geçersizse kırp
    - PAS_ASSIST=True ise: action==0 ama top-1 sığıyorsa top-1’i al
    """
    algoritma_adi = "PPO (RL) — Model"
    if pool_df.empty or capacity <= 0:
        return {
            "algoritma": algoritma_adi,
            "toplam_odul": 0.0,
            "kullanilan_kapasite": 0.0,
            "secilen_islem_sayisi": 0,
            "kapasite_doluluk_yuzdesi": 0.0,
            "ortalama_odul_per_tx": 0.0,
            "ortalama_agirlik_per_tx": 0.0,
            "tps": 0.0
        }

    # density sütunu yoksa ekle
    if "density" not in pool_df.columns:
        pool_df = pool_df.copy()
        pool_df["density"] = pool_df["value"] / (pool_df["weight"] + 1e-9)

    try:
        runner = RLPolicyRunner(model_path=model_path, vecnorm_path=vecnorm_path)
    except Exception as e:
        print(f"[RL] Model yüklenemedi ({e}). Greedy'e düşülüyor.")
        fb = solve_greedy(pool_df, capacity, n_blocks)
        fb["algoritma"] = f"{algoritma_adi} (fallback=Greedy)"
        return fb

    total_capacity = float(capacity)
    remaining_capacity = total_capacity
    total_value = 0.0
    total_weight = 0.0
    selected_count = 0

    picked_idx = set()
    # Emniyet: aşırı uzun döngüyü engelle
    for _ in range(100000):
        remaining_df = pool_df.drop(picked_idx, errors="ignore")
        if remaining_df.empty:
            break
        # Sığabilecek işlem kalmadıysa bitir
        if float(remaining_df["weight"].min()) > remaining_capacity:
            break

        # top-k listesi
        top_k = remaining_df.nlargest(K_TOP_ACTIONS, "density")
        valid_actions = 1 + len(top_k)  # 0..len(top_k)

        # obs üret ve normalize et
        obs = RLPolicyRunner.make_obs(remaining_capacity, total_capacity, remaining_df, pool_df)
        obs = runner._maybe_norm(obs)

        # model aksiyonu
        a = runner.predict_action(obs, valid_actions)

        # PAS_ASSIST: model pas dese de top-1 sığıyorsa al
        if PAS_ASSIST and a == 0 and len(top_k) > 0:
            top1 = top_k.iloc[0]
            w1 = float(top1["weight"])
            if w1 <= remaining_capacity:
                a = 1  # top-1'i seç

        if a == 0:
            # gerçek pas: hiçbir şey yapmadan devam
            pass
        else:
            chosen = top_k.iloc[a - 1]
            w = float(chosen["weight"]); v = float(chosen["value"])
            if w <= remaining_capacity:
                remaining_capacity -= w
                total_weight += w
                total_value  += v
                selected_count += 1
                picked_idx.add(chosen.name)
            # sığmıyorsa bu adımı pas geçmiş say

    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(
        total_value, total_weight, selected_count, total_capacity, n_blocks
    )

    return {
        "algoritma": algoritma_adi,
        "toplam_odul": float(total_value),
        "kullanilan_kapasite": float(total_weight),
        "secilen_islem_sayisi": int(selected_count),
        "kapasite_doluluk_yuzdesi": float(cap_util),
        "ortalama_odul_per_tx": float(avg_reward),
        "ortalama_agirlik_per_tx": float(avg_weight),
        "tps": float(tps),
    }
