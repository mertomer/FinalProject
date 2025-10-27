# eth_analyzer/simulation/algorithms.py
import random
import math
import time
import pandas as pd # Gerekli import
from deap import base, creator, tools, algorithms # Gerekli import

# config'den GA/SA parametrelerini import et (opsiyonel)
from ..config import (
    GA_POPULATION_SIZE, GA_GENERATIONS, GA_CROSSOVER_PROB, GA_MUTATION_PROB,
    SA_INITIAL_TEMP, SA_MIN_TEMP, SA_ALPHA
)

# --- YARDIMCI FONKSİYON ---
def calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity):
    """Ortak metrikleri hesaplar."""
    avg_reward_per_tx = total_value / selected_tx_count if selected_tx_count > 0 else 0
    avg_weight_per_tx = total_weight / selected_tx_count if selected_tx_count > 0 else 0
    cap_util_percent = (total_weight / capacity) * 100 if capacity > 0 else 0
    return avg_reward_per_tx, avg_weight_per_tx, cap_util_percent

# --- ALGORİTMALAR ---

def solve_random(transactions_df, capacity):
    """Algoritma 1: Rastgele Seçim (Baseline)"""
    if transactions_df.empty or capacity <= 0: return {"algoritma": "Rastgele", "toplam_odul": 0, "kullanilan_kapasite": 0, "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0, "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0}
    shuffled_df = transactions_df.sample(frac=1)
    total_value, total_weight, selected_tx_count = 0, 0, 0
    for _, tx in shuffled_df.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']; total_value += tx['value']; selected_tx_count += 1
    avg_reward, avg_weight, cap_util = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Rastgele", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight
    }

def solve_greedy(transactions_df, capacity):
    """Algoritma 2: Açgözlü (Greedy) Seçim"""
    if transactions_df.empty or capacity <= 0: return {"algoritma": "Açgözlü (Greedy)", "toplam_odul": 0, "kullanilan_kapasite": 0, "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0, "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0}
    df = transactions_df.copy()
    df['density'] = df['value'] / df['weight'] # weight > 0 kontrolü core.py'de yapıldı
    df_sorted = df.sort_values(by='density', ascending=False)
    total_value, total_weight, selected_tx_count = 0, 0, 0
    for _, tx in df_sorted.iterrows():
        if total_weight + tx['weight'] <= capacity:
            total_weight += tx['weight']; total_value += tx['value']; selected_tx_count += 1
    avg_reward, avg_weight, cap_util = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Açgözlü (Greedy)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight
    }

def solve_genetic_algorithm(transactions_df, capacity):
    """Algoritma 3: Genetik Algoritma (Yapay Zeka)"""
    if transactions_df.empty or capacity <= 0: return {"algoritma": "Genetik Algoritma (YZ)", "toplam_odul": 0, "kullanilan_kapasite": 0, "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0, "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0}
    print("\n[GA] Genetik Algoritma (YZ) çalıştırılıyor...")
    items = transactions_df[['weight', 'value']].to_dict('records')
    num_items = len(items)

    # DEAP kurulumu (Tek seferlik)
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
    toolbox.register("mutate", tools.mutFlipBit, indpb=GA_MUTATION_PROB) # config'den
    toolbox.register("select", tools.selTournament, tournsize=3)

    start_time = time.time()
    pop = toolbox.population(n=GA_POPULATION_SIZE); hof = tools.HallOfFame(1) # config'den
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda x: sum(x)/len(x) if x else 0)
    stats.register("max", lambda x: max(x) if x else 0)

    # verbose=False yaparak konsolu temiz tutalım, sadece sonucu yazdıralım
    algorithms.eaSimple(pop, toolbox, cxpb=GA_CROSSOVER_PROB, mutpb=GA_MUTATION_PROB, ngen=GA_GENERATIONS,
                        stats=stats, halloffame=hof, verbose=False) # config'den
    print(f"[GA] Genetik Algoritma {time.time() - start_time:.2f} saniyede tamamlandı.")

    best_individual = hof[0]; total_value, total_weight, selected_tx_count = 0, 0, 0
    for i in range(num_items):
        if best_individual[i] == 1:
            total_weight += items[i]['weight']; total_value += items[i]['value']; selected_tx_count += 1

    avg_reward, avg_weight, cap_util = calculate_derived_metrics(total_value, total_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Genetik Algoritma (YZ)", "toplam_odul": total_value, "kullanilan_kapasite": total_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight
    }

def solve_simulated_annealing(transactions_df, capacity):
    """Algoritma 4: Benzetilmiş Tavlama (Simulated Annealing - SA)"""
    if transactions_df.empty or capacity <= 0: return {"algoritma": "Benzetilmiş Tavlama (YZ)", "toplam_odul": 0, "kullanilan_kapasite": 0, "secilen_islem_sayisi": 0, "kapasite_doluluk_yuzdesi": 0.0, "ortalama_odul_per_tx": 0.0, "ortalama_agirlik_per_tx": 0.0}
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
    # İterasyon sayısını sınırlayalım (çok uzun sürmemesi için)
    max_iterations = 20000 # Bu değeri ayarlayabilirsiniz
    iter_count = 0

    while T > SA_MIN_TEMP and iter_count < max_iterations: # config'den
        new_solution = set(current_solution)
        # Daha dengeli komşu seçimi
        p_remove = 0.5 if len(new_solution) > 0 else 0.0
        p_add = 0.5 if len(new_solution) < num_items else 1.0 # Eğer hepsi ekliyse çıkarma zorunlu
        if len(new_solution) == 0: p_add = 1.0 # Eğer boşsa ekleme zorunlu

        action = random.uniform(0, 1)
        if action < p_remove:
            item_to_remove = random.choice(list(new_solution))
            new_solution.remove(item_to_remove)
        elif action < p_remove + p_add:
             # Sadece havuzda olmayanlardan birini eklemeyi dene
            available_to_add = list(set(range(num_items)) - new_solution)
            if available_to_add:
                item_to_add = random.choice(available_to_add)
                new_solution.add(item_to_add)
            # Eğer eklenecek yoksa bir sonrakine geç

        new_weight, new_value = get_solution_stats(new_solution)

        if new_weight <= capacity:
            delta_value = new_value - current_value
            # Sıfıra bölme hatasını engelle (T > 0 olmalı)
            if delta_value > 0 or (T > 0 and random.random() < math.exp(delta_value / T)):
                current_solution, current_weight, current_value = new_solution, new_weight, new_value
                if current_value > best_value:
                    best_solution, best_weight, best_value = current_solution, new_weight, current_value
        T *= SA_ALPHA # config'den
        iter_count += 1

    print(f"[SA] Benzetilmiş Tavlama {time.time() - start_time:.2f} saniyede tamamlandı ({iter_count} iterasyon).")
    selected_tx_count = len(best_solution)
    avg_reward, avg_weight, cap_util = calculate_derived_metrics(best_value, best_weight, selected_tx_count, capacity)
    return {
        "algoritma": "Benzetilmiş Tavlama (YZ)", "toplam_odul": best_value, "kullanilan_kapasite": best_weight,
        "secilen_islem_sayisi": selected_tx_count, "kapasite_doluluk_yuzdesi": cap_util,
        "ortalama_odul_per_tx": avg_reward, "ortalama_agirlik_per_tx": avg_weight
    }