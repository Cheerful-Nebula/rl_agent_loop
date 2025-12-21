import numpy as np
# Replace 'your_env_file' with the actual name of the file where DynamicRewardWrapper is
from train import DynamicRewardWrapper 

def run_tests():
    print("--- Starting Sanity Check ---")

    # TEST 1: The "Perfect Landing" Scenario
    # Obs: [x, y, vx, vy, angle, angular_vel, leg1, leg2]
    # We set x=0 (center), legs=1 (touching), angle=0 (upright)
    perfect_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    result = DynamicRewardWrapper.is_successful_landing(perfect_obs)
    if result == True:
        print("✅ Test 1 Passed: Perfect landing detected correctly.")
    else:
        print("❌ Test 1 Failed: Perfect landing was rejected!")

    # TEST 2: The "Hovering but Safe" Scenario
    # Everything is good, but legs aren't touching (leg1=0, leg2=0)
    hover_obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = DynamicRewardWrapper.is_successful_landing(hover_obs)
    if result == False:
        print("✅ Test 2 Passed: Hovering (no legs down) correctly marked as not landed.")
    else:
        print("❌ Test 2 Failed: Hovering was falsely marked as a landing!")

    # TEST 3: The "Crashed" Scenario
    # Legs touching, but angle is sideways (angle = 1.0 radian is ~57 degrees)
    crash_obs = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    result = DynamicRewardWrapper.is_successful_landing(crash_obs)
    if result == False:
        print("✅ Test 3 Passed: Sideways crash correctly marked as failed.")
    else:
        print("❌ Test 3 Failed: Sideways crash was falsely marked as success!")

if __name__ == "__main__":
    run_tests()