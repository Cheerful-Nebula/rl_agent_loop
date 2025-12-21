import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def run_benchmark(n_envs_list, steps_per_test=50000):
    results = []
    
    for n in n_envs_list:
        print(f"Testing n_envs = {n}...")
        
        # 1. Setup Env
        env = make_vec_env("LunarLander-v3", n_envs=n, vec_env_cls=SubprocVecEnv)
        
        # 2. Setup Model (using the scaling logic we discussed)
        n_steps = max(int(8192 // n), 16)
        model = PPO("MlpPolicy", env, device="auto", n_steps=n_steps, verbose=0)
        
        # 3. Benchmark
        start_time = time.time()
        model.learn(total_timesteps=steps_per_test)
        end_time = time.time()
        
        # 4. Calculate
        total_time = end_time - start_time
        fps = steps_per_test / total_time
        
        results.append({"n_envs": n, "fps": int(fps), "total_time": round(total_time, 2)})
        
        # Cleanup to free memory/processes for next test
        env.close()
        del model
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Test ranges: 
    # On PC, try [8, 16, 24, 32, 40]
    # On Mac, try [4, 8, 12, 16]
    test_counts = [8, 16, 24, 32, 40] 
    
    df = run_benchmark(test_counts)
    
    print("\n--- BENCHMARK RESULTS ---")
    print(df.to_string(index=False))
    
    best_n = df.loc[df['fps'].idxmax()]['n_envs']
    print(f"\nOptimal n_envs for this machine: {best_n}")