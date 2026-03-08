import argparse
import subprocess
import sys

def run_local_cycle(iteration: int, num_seeds: int):
    print(f"💻 [Local-Manager] Executing Local Training for Iteration {iteration}")

    try:
        # 1. Sequential Training
        for seed_id in range(num_seeds):
            print(f"   ▶️ Training Seed {seed_id}")
            cmd = f"python3 train.py --iteration {iteration} --seed_id {seed_id}"
            subprocess.run(cmd, shell=True, check=True)
        
        # 2. Local Aggregation
        print(f"   📊 Running Local Analysis")
        subprocess.run(f"python3 src/analysis.py --iteration {iteration} --num_seeds {num_seeds}", shell=True, check=True)
        
        print(f"✅ Local Cycle {iteration} Complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Critical Failure during local execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--num_seeds", type=int, default=3)
    args = parser.parse_args()
    
    run_local_cycle(args.iteration, args.num_seeds)