import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np

# Set up styling for IEEE format
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
})

RESULTS_DIR = "/fs/nexus-scratch/vvs22/pomdp_grasp/results"
LOGS_DIR = "/fs/nexus-scratch/vvs22/pomdp_grasp/logs"
PLOTS_DIR = "/fs/nexus-scratch/vvs22/pomdp_grasp/plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_sensitivity_curve():
    print("Generating Sensitivity Curve...")
    baseline_path = os.path.join(RESULTS_DIR, "noise_sensitivity.json")
    noisy_path = os.path.join(RESULTS_DIR, "noisy_policy_noise_sensitivity.json")
    
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    with open(noisy_path, 'r') as f:
        noisy_data = json.load(f)
        
    noise_stds = baseline_data["noise_std"]
    
    # Extract success rates in the same order as noise_stds
    baseline_success = [baseline_data["success_rate"][str(std)] for std in noise_stds]
    noisy_success = [noisy_data["success_rate"][str(std)] for std in noise_stds]
    
    # Convert noise std to cm for better readability
    noise_cm = [std * 100 for std in noise_stds]
    
    plt.figure()
    plt.plot(noise_cm, baseline_success, marker='o', linestyle='-', linewidth=2, color='#1f77b4', label='Baseline Policy (0cm Training Noise)')
    plt.plot(noise_cm, noisy_success, marker='s', linestyle='--', linewidth=2, color='#ff7f0e', label='Fine-Tuned Policy (3cm Training Noise)')
    
    plt.axvline(x=3.0, color='gray', linestyle=':', alpha=0.7, label='3cm Training Bound')
    
    plt.xlabel('Observation Noise Std Dev (cm)')
    plt.ylabel('Lift Success Rate (%)')
    plt.title('Policy Sensitivity to Observation Noise')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(40, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sensitivity_curve.png"), format='png', bbox_inches='tight')
    plt.close()

def parse_rewards(log_path):
    rewards = []
    # Regex to match the reward line
    pattern = re.compile(r"Episode_Reward/lifting_object:\s+([0-9\.]+)")
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}")
        return []
        
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                rewards.append(float(match.group(1)))
    return rewards

def plot_training_stability():
    print("Generating Training Stability Plot...")
    # Update these filenames if necessary based on your logs directory
    failed_log = os.path.join(LOGS_DIR, "train_noisy_lift_6815944.log") # The 10cm collapsed run
    success_log = os.path.join(LOGS_DIR, "train_noisy_lift_6821885.log") # The 3cm successful run
    
    failed_rewards = parse_rewards(failed_log)
    success_rewards = parse_rewards(success_log)
    
    # Smooth the curves using a moving average
    def smooth(y, box_pts=10):
        if len(y) < box_pts: return y
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
        
    plt.figure()
    
    if success_rewards:
        plt.plot(smooth(success_rewards, 20), color='#2ca02c', linewidth=2, label='3cm Max Noise (Stable)')
    if failed_rewards:
        plt.plot(smooth(failed_rewards, 20), color='#d62728', linewidth=2, alpha=0.8, label='10cm Max Noise (Collapsed)')
        
    plt.xlabel('Fine-Tuning Iterations')
    plt.ylabel('Episode Reward (Lifting)')
    plt.title('Catastrophic Forgetting vs. Stable Fine-Tuning')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_stability.png"), format='png', bbox_inches='tight')
    plt.close()

def plot_particle_filter():
    print("Generating Particle Filter Convergence Plot...")
    observations = [0, 3, 5, 10]
    mean_error = [0.0724 * 100, 0.0298 * 100, 0.0183 * 100, 0.0047 * 100] # Convert to cm
    entropy = [-4.42, -7.08, -7.54, -8.35]
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color1 = '#1f77b4'
    ax1.set_xlabel('Sequential Observations')
    ax1.set_ylabel('Mean Position Error (cm)', color=color1)
    ax1.plot(observations, mean_error, marker='o', color=color1, linewidth=2, label='Position Error')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'
    ax2.set_ylabel('Belief Entropy (nats)', color=color2)
    ax2.plot(observations, entropy, marker='s', color=color2, linewidth=2, linestyle='--', label='Belief Entropy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Particle Filter Belief Convergence')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "particle_filter_convergence.png"), format='png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_sensitivity_curve()
    plot_training_stability()
    plot_particle_filter()
    print(f"\nAll plots successfully generated and saved to: {PLOTS_DIR}")
