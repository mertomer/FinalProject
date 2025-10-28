import os
import random
import math
import time
import pandas as pd

import numpy as np

DEAP_AVAILABLE = True
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    DEAP_AVAILABLE = False
    print("UYARI: 'deap' kütüphanesi bulunamadı. GA için yerel (DEAP'siz) çözüm çalışacak.")

STABLE_BASELINES_AVAILABLE = True
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("UYARI: 'stable-baselines3' kütüphanesi bulunamadı. RL modeli kullanılamayacak.")

try:
    from ..config import (
        GA_POPULATION_SIZE, GA_GENERATIONS, GA_CROSSOVER_PROB, GA_MUTATION_PROB,
        SA_INITIAL_TEMP, SA_MIN_TEMP, SA_ALPHA,
        RL_MODEL_PATH, RL_VECNORMALIZE_PATH,
        DEFAULT_START_BLOCK
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
    RL_MODEL_PATH = None
    RL_VECNORMALIZE_PATH = None
    DEFAULT_START_BLOCK = 20866020

try:
    from ..rl.environment import MempoolEnv
except ImportError:
    MempoolEnv = None
    print("UYARI: RL ortamı import edilemedi. RL modeli kullanılamayacak.")

_RL_MODEL = None
_RL_MODEL_LOAD_ERROR = None

# --- YARDIMCI FONKSİYON (GÜNCELLENDİ) ---
def calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity, n_blocks):
    """
    Yardımcı fonksiyon: Ortak metrikleri (TPS dahil) hesaplar.
    """
    avg_reward_per_tx = total_value / selected_tx_count if selected_tx_count > 0 else 0
    avg_weight_per_tx = total_weight / selected_tx_count if selected_tx_count > 0 else 0
    cap_util_percent = (total_weight / capacity) * 100 if capacity > 0 else 0
    
    # TPS Hesaplaması (Ortalama blok süresi ~12 saniye)
    estimated_duration_seconds = n_blocks * 12 
    tps = selected_tx_count / estimated_duration_seconds if estimated_duration_seconds > 0 else 0
    
    # Hesaplanan tüm metrikleri döndür
    return avg_reward_per_tx, avg_weight_per_tx, cap_util_percent, tps


def _empty_result(algorithm_name):
    return {
        "algoritma": algorithm_name,
        "toplam_odul": 0,
        "kullanilan_kapasite": 0,
        "secilen_islem_sayisi": 0,
        "kapasite_doluluk_yuzdesi": 0.0,
        "ortalama_odul_per_tx": 0.0,
        "ortalama_agirlik_per_tx": 0.0,
        "tps": 0.0
    }


def _load_rl_model():
    global _RL_MODEL, _RL_MODEL_LOAD_ERROR

    if _RL_MODEL is not None:
        return _RL_MODEL

    if _RL_MODEL_LOAD_ERROR is not None:
        return None

    if not STABLE_BASELINES_AVAILABLE:
        _RL_MODEL_LOAD_ERROR = "stable-baselines3 eksik"
        return None

    if RL_MODEL_PATH is None or not os.path.exists(RL_MODEL_PATH):
        _RL_MODEL_LOAD_ERROR = f"Model dosyası bulunamadı: {RL_MODEL_PATH}"
        print(f"[RL] Model yolu bulunamadı: {RL_MODEL_PATH}")
        return None

    try:
        _RL_MODEL = PPO.load(RL_MODEL_PATH)
    except Exception as exc:
        _RL_MODEL_LOAD_ERROR = str(exc)
        print(f"[RL] Model yüklenemedi: {exc}")
        return None

    return _RL_MODEL


def solve_random(transactions_df, capacity, n_blocks):
    """Algoritma 1: Rastgele Seçim (Baseline)"""
    # Boş havuz veya sıfır kapasite kontrolü
    if transactions_df.empty or capacity <= 0:
        return _empty_result("Rastgele")
        
    shuffled_df = transactions_df.sample(frac=1)
    total_value, total_weight, selected_tx_count = 0, 0, 0
    
    for _, tx in shuffled_df.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1
            
    # Güncellenmiş metrik hesaplaması (n_blocks ile)
    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity, n_blocks) 
    
    return {
        "algoritma": "Rastgele", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps 
    }

def solve_greedy(transactions_df, capacity, n_blocks):
    """Algoritma 2: Açgözlü (Greedy) Seçim"""
    if transactions_df.empty or capacity <= 0:
        return _empty_result("Açgözlü (Greedy)")
        
    df = transactions_df.copy()
   
    df['density'] = df['value'] / (df['weight'] + 1e-9) 
    df_sorted = df.sort_values(by='density', ascending=False)
    total_value, total_weight, selected_tx_count = 0, 0, 0
    
    for _, tx in df_sorted.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']
            total_value += tx['value']
            selected_tx_count += 1
            
    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity, n_blocks)
    
    return {
        "algoritma": "Açgözlü (Greedy)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps 
    }

def solve_genetic_algorithm(transactions_df, capacity, n_blocks):
    """Algoritma 3: Genetik Algoritma (Yapay Zeka)"""
    if transactions_df.empty or capacity <= 0:
        return _empty_result("Genetik Algoritma (YZ)")
        
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

        # Başlangıç popülasyonu
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
        total_weight = 0
        selected_tx_count = 0
        for i, bit in enumerate(best_genome):
            if bit == 1:
                total_weight += items[i]['weight']
                selected_tx_count += 1

        avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity, n_blocks)
        return {
            "algoritma": "Genetik Algoritma (YZ)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
            "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
            "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
            "tps": tps 
        }

    print("\n[GA] Genetik Algoritma (YZ) çalıştırılıyor...")
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

    toolbox.register("evaluate", evalKnapsack); toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=GA_MUTATION_PROB) 
    toolbox.register("select", tools.selTournament, tournsize=3)

    start_time = time.time()
    pop = toolbox.population(n=GA_POPULATION_SIZE); hof = tools.HallOfFame(1) 
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x)/len(x) if x else 0)
    stats.register("max", lambda x: max(x) if x else 0)

    algorithms.eaSimple(pop, toolbox, cxpb=GA_CROSSOVER_PROB, mutpb=GA_MUTATION_PROB, ngen=GA_GENERATIONS,
                        stats=stats, halloffame=hof, verbose=False) 
    print(f"[GA] Genetik Algoritma {time.time() - start_time:.2f} saniyede tamamlandı.")

    best_individual = hof[0]; total_value, total_weight, selected_tx_count = 0, 0, 0
    for i in range(num_items):
        if best_individual[i] == 1:
            total_weight += items[i]['weight']; total_value += items[i]['value']; selected_tx_count += 1

    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity, n_blocks)
    
    return {
        "algoritma": "Genetik Algoritma (YZ)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps 
    }

def solve_simulated_annealing(transactions_df, capacity, n_blocks):
    """Algoritma 4: Benzetilmiş Tavlama (Simulated Annealing - SA)"""
    if transactions_df.empty or capacity <= 0:
        return _empty_result("Benzetilmiş Tavlama (YZ)")
        
    print("\n[SA] Benzetilmiş Tavlama (YZ) çalıştırılıyor...")
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

    T = SA_INITIAL_TEMP # config'den
    current_solution, current_weight, current_value = create_initial_solution()
    best_solution, best_value, best_weight = current_solution, current_value, current_weight

    start_time = time.time()
    max_iterations = 20000 
    iter_count = 0

    while T > SA_MIN_TEMP and iter_count < max_iterations: 
        new_solution = set(current_solution)
        
        p_remove = 0.5 if len(new_solution) > 0 else 0.0
        p_add = 0.5 if len(new_solution) < num_items else 1.0 
        if len(new_solution) == 0: p_add = 1.0 

        action = random.uniform(0, 1)
        if action < p_remove:
            item_to_remove = random.choice(list(new_solution))
            new_solution.remove(item_to_remove)
        elif action < p_remove + p_add:
             available_to_add = list(set(range(num_items)) - new_solution)
             if available_to_add:
                 item_to_add = random.choice(available_to_add)
                 new_solution.add(item_to_add)

        new_weight, new_value = get_solution_stats(new_solution)

        if new_weight <= capacity:
            delta_value = new_value - current_value
            if delta_value > 0 or (T > 1e-9 and random.random() < math.exp(delta_value / T)): # T > 0 kontrolü
                current_solution, current_weight, current_value = new_solution, new_weight, new_value
                if current_value > best_value:
                    best_solution, best_weight, best_value = current_solution, new_weight, current_value
        T *= SA_ALPHA 
        iter_count += 1

    print(f"[SA] Benzetilmiş Tavlama {time.time() - start_time:.2f} saniyede tamamlandı ({iter_count} iterasyon).")
    selected_tx_count = len(best_solution)
    
    # Güncellenmiş metrik hesaplaması (n_blocks ile)
    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(best_value, best_weight, selected_tx_count, capacity, n_blocks)
    
    return {
        "algoritma": "Benzetilmiş Tavlama (YZ)", "toplam_odul": best_value, "kullanilan_kapasite": best_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps
    }


def solve_rl_policy(transactions_df, capacity, n_blocks, start_block=None):
    """Algoritma 5: Öğrenilmiş PPO tabanlı RL politikası."""

    algorithm_name = "PPO (RL)"

    if transactions_df.empty or capacity <= 0:
        return _empty_result(algorithm_name)

    if not STABLE_BASELINES_AVAILABLE or MempoolEnv is None:
        print("[RL] Gerekli RL bağımlılıkları eksik. RL algoritması atlanıyor.")
        return _empty_result(algorithm_name)

    model = _load_rl_model()
    if model is None:
        return _empty_result(algorithm_name)

    # Havuzu hazırlayalım (RL ortamı yoğunluk sütununa ihtiyaç duyuyor)
    prepared_df = transactions_df.copy().reset_index(drop=True)
    if 'density' not in prepared_df.columns:
        prepared_df['density'] = prepared_df['value'] / (prepared_df['weight'] + 1e-9)

    env_kwargs = dict(
        start_block=start_block if start_block is not None else DEFAULT_START_BLOCK,
        n_blocks=n_blocks,
        pool_df=prepared_df,
        total_capacity=capacity
    )

    try:
        base_env = MempoolEnv(**env_kwargs)
    except Exception as exc:
        print(f"[RL] Ortam oluşturulamadı: {exc}")
        return _empty_result(algorithm_name)

    vec_env = DummyVecEnv([lambda: base_env])

    if RL_VECNORMALIZE_PATH and os.path.exists(RL_VECNORMALIZE_PATH):
        try:
            vec_env = VecNormalize.load(RL_VECNORMALIZE_PATH, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        except Exception as exc:
            print(f"[RL] VecNormalize yüklenemedi ({exc}). Normalize edilmemiş ortam kullanılacak.")
            vec_env = DummyVecEnv([lambda: base_env])

    try:
        model.set_env(vec_env)
        reset_result = vec_env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result

        done = np.array([False])
        safety_counter = 0
        max_steps = len(prepared_df) + 5

        while not bool(done[0]):
            action, _ = model.predict(obs, deterministic=True)
            step_result = vec_env.step(action)

            if len(step_result) == 4:
                obs, _rewards, dones, _infos = step_result
                done = np.array(dones, dtype=bool)
            elif len(step_result) == 5:
                obs, _rewards, terminated, truncated, _infos = step_result
                done = np.array(terminated, dtype=bool) | np.array(truncated, dtype=bool)
            else:
                raise ValueError("VecEnv.step beklenmeyen çıktı döndürdü")

            safety_counter += 1
            if safety_counter > max_steps:
                print("[RL] Güvenlik sınırı aşıldı, döngü sonlandırıldı.")
                break

    except Exception as exc:
        print(f"[RL] Politika yürütülürken hata oluştu: {exc}")
        return _empty_result(algorithm_name)

    # Temel ortam örneğini alın (VecNormalize kullanılıyorsa iç ortamı çek)
    underlying_env = vec_env
    if isinstance(vec_env, VecNormalize):
        underlying_env = vec_env.venv

    env_instance = underlying_env.envs[0]

    selected_tx_count = len(env_instance.selected_tx_indices)
    total_value = env_instance.current_total_reward
    total_weight = env_instance.total_capacity - env_instance.remaining_capacity

    avg_reward, avg_weight, cap_util, tps = calculate_derived_metrics(
        total_value, total_weight, selected_tx_count, capacity, n_blocks
    )

    return {
        "algoritma": algorithm_name,
        "toplam_odul": total_value,
        "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count,
        "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward,
        "ortalama_agirlik_per_tx": avg_weight,
        "tps": tps
    }
