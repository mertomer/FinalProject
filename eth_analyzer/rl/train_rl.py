# train_rl.py (Proje Kök Dizininde)
import os
import sys
import pandas as pd
import time
import argparse

# Proje kök dizinini Python yoluna ekle
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Gerekli importlar
from eth_analyzer.rl.environment import MempoolEnv          # Oluşturduğumuz ortam
from eth_analyzer.simulation.algorithms import solve_greedy # Kıyaslama için Greedy
from eth_analyzer.config import DEFAULT_START_BLOCK, SIMULATION_MIN_WINDOW

# RL kütüphanesi
try:
    from stable_baselines3 import PPO # Popüler bir algoritma, DQN veya A2C de olabilir
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        CallbackList,
        EvalCallback,
        StopTrainingOnNoModelImprovement,
        StopTrainingOnRewardThreshold,
    )
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("HATA: 'stable-baselines3' veya bağımlılıkları bulunamadı.")
    print("Lütfen 'pip install stable-baselines3[extra]' komutunu çalıştırın.")
    sys.exit(1)

def evaluate_agent(model, env, num_episodes=10):
    """Eğitilmiş ajanı değerlendirir ve ortalama ödülü döndürür."""
    total_rewards = 0
    total_tx_count = 0
    start_time = time.time()
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True) # En iyi aksiyonu seç
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        total_rewards += info.get('current_total_reward', 0) # Bölüm sonundaki toplam ödül
        total_tx_count += info.get('selected_count', 0)
    
    avg_reward = total_rewards / num_episodes
    avg_tx_count = total_tx_count / num_episodes
    duration = time.time() - start_time
    print(f"Değerlendirme ({num_episodes} bölüm) {duration:.2f} saniyede tamamlandı.")
    return avg_reward, avg_tx_count

def linear_schedule(initial_value: float):
    """SB3 ile lineer öğrenme oranı takvimi."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


def make_env(start_block: int, n_blocks: int, k_top_actions: int, seed: int = 0):
    """SubprocVecEnv uyumlu ortam kurucu."""
    def _init():
        env = MempoolEnv(start_block=start_block, n_blocks=n_blocks, k_top_actions=k_top_actions)
        env = Monitor(env)
        return env
    return _init


def parse_args():
    parser = argparse.ArgumentParser(description="RL mempool eğitim betiği")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--n-envs", type=int, default=4, help="Paralel ortam sayısı")
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "linear"], default="linear")
    parser.add_argument("--policy-width", type=int, default=128, help="Policy ağındaki katman genişliği")
    parser.add_argument("--policy-depth", type=int, default=2, help="Policy ağındaki gizli katman sayısı")
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--reward-threshold", type=float, default=None, help="Bu ödül ortalamasına ulaşınca erken durdur")
    parser.add_argument("--start-block", type=int, default=DEFAULT_START_BLOCK)
    parser.add_argument("--n-blocks", type=int, default=SIMULATION_MIN_WINDOW)
    parser.add_argument("--k-top-actions", type=int, default=10)
    return parser.parse_args()


def main():
    print("=" * 60)
    print("Pekiştirmeli Öğrenme (RL) Ajansı Eğitimi Başlatılıyor")
    print("=" * 60)

    # --- Argümanlar ---
    args = parse_args()

    # --- Ayarlar ---
    N_BLOCKS = args.n_blocks
    K_TOP_ACTIONS = args.k_top_actions
    TOTAL_TIMESTEPS = args.total_timesteps
    N_ENVS = max(1, args.n_envs)
    MODEL_SAVE_PATH = os.path.join(project_root, "results", "rl_mempool_model_ppo")
    TENSORBOARD_LOG = os.path.join(project_root, "results", "tb")

    # --- Ortamı Oluştur ---
    print(f"\n[1] RL Ortamı ({N_BLOCKS} blok, K={K_TOP_ACTIONS}, n_envs={N_ENVS}) oluşturuluyor...")
    try:
        # Çoklu süreçli ortam
        if N_ENVS > 1:
            env = SubprocVecEnv([
                make_env(start_block=args.start_block, n_blocks=N_BLOCKS, k_top_actions=K_TOP_ACTIONS, seed=i)
                for i in range(N_ENVS)
            ])
        else:
            env = DummyVecEnv([make_env(start_block=args.start_block, n_blocks=N_BLOCKS, k_top_actions=K_TOP_ACTIONS, seed=0)])

        # Gözlem/ödül normalizasyonu
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Ortamın Gymnasium standartlarına uygunluğunu kontrol et (önemli!)
        # check_env(env.envs[0])
        print("✅ Ortam başarıyla oluşturuldu.")
    except Exception as e:
        print(f"❌ Ortam oluşturulurken hata: {e}")
        return

    # --- Modeli Oluştur ve Eğit ---
    print(f"\n[2] PPO Modeli oluşturuluyor ve {TOTAL_TIMESTEPS} adım eğitiliyor...")
    # Öğrenme oranı takvimi
    lr = linear_schedule(args.learning_rate) if args.lr_schedule == "linear" else args.learning_rate

    # Politika ağı mimarisi
    hidden_layers = [args.policy_width] * args.policy_depth
    policy_kwargs = dict(net_arch=dict(pi=hidden_layers, vf=hidden_layers))

    # PPO hiperparametreleri
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=lr,
        tensorboard_log=TENSORBOARD_LOG,
        policy_kwargs=policy_kwargs,
    ) 
    
    start_train_time = time.time()
    try:
        # Eğitim süresince canlı ETA ve hız bilgisi basan callback
        class ProgressCallback(BaseCallback):
            def __init__(self, total_timesteps: int, log_interval_sec: float = 5.0):
                super().__init__()
                self.total_timesteps = total_timesteps
                self.log_interval_sec = log_interval_sec
                self._start_time = None
                self._last_log_time = 0.0

            def _on_training_start(self) -> None:
                self._start_time = time.time()
                self._last_log_time = self._start_time

            def _on_step(self) -> bool:
                now = time.time()
                if now - self._last_log_time >= self.log_interval_sec:
                    done_steps = int(self.model.num_timesteps)
                    elapsed = max(now - self._start_time, 1e-6)
                    steps_per_sec = done_steps / elapsed
                    remaining = max(self.total_timesteps - done_steps, 0)
                    eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else float('inf')
                    percent = (done_steps / self.total_timesteps) * 100 if self.total_timesteps > 0 else 0.0
                    print(
                        f"İlerleme: {done_steps}/{self.total_timesteps} (%{percent:.1f}), "
                        f"hız ~{steps_per_sec:.0f} adım/sn, kalan ~{eta_sec:.1f} sn"
                    )
                    self._last_log_time = now
                return True

            def _on_training_end(self) -> None:
                total_elapsed = time.time() - self._start_time if self._start_time else 0.0
                if total_elapsed > 0:
                    avg_speed = self.total_timesteps / total_elapsed
                    print(f"Eğitim tamamlandı. Toplam süre: {total_elapsed:.2f} sn, ort. hız ~{avg_speed:.0f} adım/sn")

        # Checkpoint ve ilerleme callback'lerini birlikte kullan
        ckpt_dir = os.path.join(project_root, "results", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=5000,
            save_path=ckpt_dir,
            name_prefix="ppo_mempool",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # Değerlendirme ortamı (aynı dinamiklerle)
        eval_env = DummyVecEnv([make_env(start_block=args.start_block, n_blocks=N_BLOCKS, k_top_actions=K_TOP_ACTIONS, seed=123)])
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Eğitimdeki normalizasyon istatistiklerini eval'e kopyala
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms

        # İyileşme yoksa erken durdurma (opsiyonel ödül eşiğiyle)
        stop_no_improve = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,
            min_evals=2,
            verbose=1,
        )
        if args.reward_threshold is not None:
            stop_on_reward = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1)
            early_stops = CallbackList([stop_no_improve, stop_on_reward])
        else:
            early_stops = stop_no_improve

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(project_root, "results", "best_model"),
            log_path=os.path.join(project_root, "results", "eval_logs"),
            eval_freq=max(args.eval_freq // N_ENVS, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
            callback_after_eval=early_stops,
        )

        callback_list = CallbackList([ProgressCallback(TOTAL_TIMESTEPS), checkpoint_cb, eval_cb])

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            progress_bar=True,   # rich tabanlı progress bar
        )
        duration = time.time() - start_train_time
        print(f"✅ Eğitim {duration:.2f} saniyede tamamlandı.")
        
        # Eğitilmiş modeli kaydet
        model.save(MODEL_SAVE_PATH)
        print(f"💾 Model şuraya kaydedildi: {MODEL_SAVE_PATH}.zip")

        # VecNormalize istatistiklerini kaydet
        vecnorm_path = os.path.join(project_root, "results", "vecnormalize.pkl")
        env.save(vecnorm_path)
        print(f"💾 VecNormalize istatistikleri kaydedildi: {vecnorm_path}")
        
    except KeyboardInterrupt:
        # Kesinti durumunda o ana kadarki modeli kaydet
        print("\n⏹ Eğitim kesildi (KeyboardInterrupt). Mevcut model kaydediliyor...")
        try:
            model.save(MODEL_SAVE_PATH)
            print(f"💾 Geçici model kaydedildi: {MODEL_SAVE_PATH}.zip")
        except Exception as save_err:
            print(f"⚠️ Model kaydedilemedi: {save_err}")
        return
    except Exception as e:
        print(f"❌ Eğitim sırasında hata: {e}")
        return

    # --- Değerlendirme ve Kıyaslama ---
    print("\n[3] Eğitilmiş RL Ajanı değerlendiriliyor...")
    # Değerlendirme env (normalizasyon ile)
    raw_eval_env = DummyVecEnv([make_env(start_block=args.start_block, n_blocks=N_BLOCKS, k_top_actions=K_TOP_ACTIONS, seed=999)])
    vecnorm_path = os.path.join(project_root, "results", "vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, raw_eval_env)
        eval_env.training = False
    else:
        eval_env = raw_eval_env

    # Vektör sarmalayıcı olmadan evaluation helper kullanmak için tek env'i içeriden al
    single_env = MempoolEnv(start_block=args.start_block, n_blocks=N_BLOCKS, k_top_actions=K_TOP_ACTIONS)
    rl_avg_reward, rl_avg_tx_count = evaluate_agent(model, single_env, num_episodes=20)
    print(f"🤖 RL Ajanı Ortalama Ödül: {rl_avg_reward:,.0f}")
    print(f"🤖 RL Ajanı Ortalama Seçilen İşlem: {rl_avg_tx_count:.1f}")

    print("\n[4] Greedy Algoritması referans olarak çalıştırılıyor...")
    # Greedy için havuzu alalım (eval_env'nin içinden alabiliriz)
    pool_df = eval_env.full_pool_df
    capacity = eval_env.total_capacity
    greedy_result = solve_greedy(pool_df, capacity, n_blocks=N_BLOCKS) # Greedy'ye de n_blocks gerekliydi
    
    print(f"📊 Greedy Ortalama Ödül: {greedy_result['toplam_odul']:,.0f}")
    print(f"📊 Greedy Ortalama Seçilen İşlem: {greedy_result['secilen_islem_sayisi']}")

    # --- Sonuç ---
    print("\n" + "=" * 60)
    print("KIYASLAMA SONUCU")
    print("=" * 60)
    if rl_avg_reward > greedy_result['toplam_odul']:
        fark = rl_avg_reward - greedy_result['toplam_odul']
        yuzde = (fark / greedy_result['toplam_odul']) * 100 if greedy_result['toplam_odul'] > 0 else float('inf')
        print(f"✅ RL Ajanı, Greedy'den % {yuzde:.2f} daha iyi performans gösterdi!")
    else:
        print(f"ℹ️ Greedy, RL Ajanından daha iyi (veya eşit) performans gösterdi.")
    print("Not: Daha iyi sonuçlar için TOTAL_TIMESTEPS artırılabilir.")

if __name__ == "__main__":
    main()