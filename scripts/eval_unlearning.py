"""Evaluate unlearning: 3 models × 12 clips, save CSV + plots.

Usage:
    uv run python scripts/eval_unlearning.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "g1" / "samples" / "npz"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp1_unlearning_validity"

MODELS = {
    "Base (no unlearning)": {
        "path": "checkpoints/base_model.pt",
        "run_dir": "base",
    },
    "Model A (λ=8.0)": {
        "path": "checkpoints/unlearned_lambda8.pt",
        "run_dir": "lambda8",
    },
    "Model B (λ=16.0)": {
        "path": "checkpoints/unlearned_lambda16.pt",
        "run_dir": "lambda16",
    },
}

# Clip names sorted by trajectory index (matching the motion library order)
CLIP_NAMES = [
    "dance1_subject1",
    "fallAndGetUp1_subject1",
    "fight1_subject2",
    "fightAndSports1_subject1",
    "flip_090_001__A304",
    "flip_090_002__A304",
    "flip_090_003__A304_M",
    "flip_360_009__A416",
    "flip_360_011__A416",
    "run1_subject2",
    "sprint1_subject2",
    "walk1_subject1",
]

TARGET_CLIP_INDICES = {2, 3}  # fight1_subject2, fightAndSports1_subject1

TASK_ID = "Wbc-G1"
NUM_ENVS = 64        # parallel envs for evaluation
NUM_EPISODES = 20    # episodes per clip per model (each env runs this many)
MAX_STEPS_PER_EP = 1000  # max steps per episode (10s @ 50Hz = 500, use 1000 for safety)


def prepare_task_and_env_cfg(motion_file: Path):
    """Register the task and return env_cfg, agent_cfg, runner_cls."""
    from wbc_mjlab.tasks import prepare_wbc_run
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    prepare_wbc_run(task_id=TASK_ID)

    from mjlab.tasks.registry import _REGISTRY
    env_cfg = _REGISTRY[TASK_ID].env_cfg

    # Set motion file for this clip
    motion_cmd = env_cfg.commands.get("motion")
    if motion_cmd is not None:
        motion_cmd.motion_file = str(motion_file)
        # Disable adaptive RSI for evaluation — use uniform
        motion_cmd.rsi.sampling_mode = "uniform"
        # Disable assistive wrench for fair evaluation
        motion_cmd.assistive_wrench_enabled = False

    # Reduce num_envs for evaluation
    env_cfg.scene.num_envs = NUM_ENVS

    agent_cfg = load_rl_cfg(TASK_ID)
    runner_cls = load_runner_cls(TASK_ID)

    return env_cfg, agent_cfg, runner_cls


def evaluate_model_on_clip(
    model_path: Path,
    clip_npz: Path,
    device: str = "cuda:0",
) -> dict:
    """Run evaluation and return metrics dict."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    env_cfg, agent_cfg_dict, runner_cls = prepare_task_and_env_cfg(clip_npz)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.clip_actions)

    # Load model
    runner = runner_cls(env, asdict(agent_cfg_dict), device=device)
    runner.load(str(model_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Collect metrics
    episode_rewards = []
    episode_lengths = []
    episode_tracking_rewards = []

    # Get motion command to track trajectory IDs
    motion_cmd = env.unwrapped.command_manager.get_term("motion")

    # Run evaluation episodes
    num_episodes_collected = 0
    current_ep_reward = torch.zeros(NUM_ENVS, device=device)
    current_ep_tracking = torch.zeros(NUM_ENVS, device=device)
    current_ep_len = torch.zeros(NUM_ENVS, device=device)

    obs, _ = env.reset()

    for step in range(MAX_STEPS_PER_EP):
        with torch.no_grad():
            actions = policy(obs)

        obs, rewards, dones, extras = env.step(actions)

        current_ep_reward += rewards
        current_ep_len += 1

        # Get tracking reward (sum of motion_* terms)
        reward_manager = env.unwrapped.reward_manager
        tracking_sum = torch.zeros(NUM_ENVS, device=device)
        weight_sum = 0.0
        for term_idx, (name, term_cfg) in enumerate(
            zip(reward_manager._term_names, reward_manager._term_cfgs, strict=False)
        ):
            if name.startswith("motion_") and term_cfg.weight > 0:
                tracking_sum += reward_manager._step_reward[:, term_idx]
                weight_sum += term_cfg.weight
        mean_tracking = tracking_sum / max(weight_sum, 1e-6)
        current_ep_tracking += mean_tracking

        # Check for done
        if dones.any():
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            for env_idx in done_indices:
                episode_rewards.append(current_ep_reward[env_idx].item())
                episode_lengths.append(current_ep_len[env_idx].item())
                episode_tracking_rewards.append(
                    current_ep_tracking[env_idx].item() / max(current_ep_len[env_idx].item(), 1)
                )
                num_episodes_collected += 1

            # Reset counters for done envs
            current_ep_reward[done_indices] = 0.0
            current_ep_tracking[done_indices] = 0.0
            current_ep_len[done_indices] = 0.0

        if num_episodes_collected >= NUM_EPISODES:
            break

    env.close()

    # Trim to NUM_EPISODES
    episode_rewards = episode_rewards[:NUM_EPISODES]
    episode_lengths = episode_lengths[:NUM_EPISODES]
    episode_tracking_rewards = episode_tracking_rewards[:NUM_EPISODES]

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_tracking": float(np.mean(episode_tracking_rewards)),
        "std_tracking": float(np.std(episode_tracking_rewards)),
        "success_rate": float(np.mean([l >= 490 for l in episode_lengths])),  # 10s = 500 steps, 490 = 9.8s
        "n_episodes": len(episode_rewards),
    }


def run_all_evaluations():
    """Evaluate all models on all clips and save results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    # Results storage
    results = []

    for model_name, model_info in MODELS.items():
        model_path = PROJECT_ROOT / model_info["path"]
        if not model_path.exists():
            print(f"[WARN] Model not found: {model_path}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] Evaluating: {model_name}")
        print(f"[INFO] Checkpoint: {model_path}")
        print(f"{'='*60}")

        for clip_idx, clip_name in enumerate(CLIP_NAMES):
            clip_npz = DATA_DIR / f"{clip_name}.npz"
            if not clip_npz.exists():
                print(f"  [WARN] Clip not found: {clip_npz}, skipping.")
                continue

            is_target = clip_idx in TARGET_CLIP_INDICES
            marker = " [TARGET]" if is_target else ""
            print(f"  [{clip_idx:2d}/12] {clip_name}{marker}...", end=" ", flush=True)

            metrics = evaluate_model_on_clip(model_path, clip_npz, device=device)

            results.append({
                "model": model_name,
                "clip_name": clip_name,
                "clip_index": clip_idx,
                "is_target": is_target,
                **metrics,
            })

            print(
                f"reward={metrics['mean_reward']:.2f}±{metrics['std_reward']:.2f} "
                f"len={metrics['mean_length']:.0f} "
                f"track={metrics['mean_tracking']:.3f} "
                f"success={metrics['success_rate']:.0%}"
            )

    # Save CSV
    csv_path = OUTPUT_DIR / "results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[INFO] Results saved to: {csv_path}")

    # Generate plots
    generate_plots(results)
    print(f"[INFO] Plots saved to: {OUTPUT_DIR}")


def generate_plots(results: list[dict]):
    """Generate comparison plots."""
    if not results:
        print("[WARN] No results to plot.")
        return

    # Organize data
    model_names = list(MODELS.keys())
    clip_labels = [n.replace("_", " ") for n in CLIP_NAMES]
    n_clips = len(CLIP_NAMES)
    n_models = len(model_names)

    # Build matrices
    tracking_matrix = np.zeros((n_models, n_clips))
    reward_matrix = np.zeros((n_models, n_clips))
    length_matrix = np.zeros((n_models, n_clips))
    success_matrix = np.zeros((n_models, n_clips))

    for r in results:
        m_idx = model_names.index(r["model"])
        c_idx = r["clip_index"]
        tracking_matrix[m_idx, c_idx] = r["mean_tracking"]
        reward_matrix[m_idx, c_idx] = r["mean_reward"]
        length_matrix[m_idx, c_idx] = r["mean_length"]
        success_matrix[m_idx, c_idx] = r["success_rate"]

    # Short clip names for x-axis
    short_names = [
        "dance", "fall&getUp", "fight1", "fight&sports",
        "flip90-1", "flip90-2", "flip90-3",
        "flip360-1", "flip360-2",
        "run", "sprint", "walk"
    ]

    colors = ["#2196F3", "#FF9800", "#F44336"]
    x = np.arange(n_clips)
    width = 0.25

    # Highlight target clips
    target_mask = np.array([i in TARGET_CLIP_INDICES for i in range(n_clips)])

    # ── Figure 1: Tracking reward comparison ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment 1: Unlearning Effectiveness Validation", fontsize=16, fontweight="bold")

    # 1a: Tracking reward per clip
    ax = axes[0, 0]
    for i, name in enumerate(model_names):
        ax.bar(x + i * width, tracking_matrix[i], width, label=name, color=colors[i], alpha=0.85)
    ax.set_xlabel("Motion Clip")
    ax.set_ylabel("Mean Tracking Reward (normalized)")
    ax.set_title("(a) Per-Clip Tracking Reward")
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="threshold")
    # Highlight target region
    for i in range(n_clips):
        if target_mask[i]:
            ax.axvspan(i - 0.4, i + 0.4 + 2*width, alpha=0.1, color="red")
    ax.text(2.5, 0.95, "TARGET", color="red", fontsize=8, ha="center", fontweight="bold")

    # 1b: Episode length per clip
    ax = axes[0, 1]
    for i, name in enumerate(model_names):
        ax.bar(x + i * width, length_matrix[i], width, label=name, color=colors[i], alpha=0.85)
    ax.set_xlabel("Motion Clip")
    ax.set_ylabel("Mean Episode Length (steps)")
    ax.set_title("(b) Per-Clip Episode Length")
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.axhline(y=500, color="gray", linestyle="--", alpha=0.5, label="max (10s)")
    for i in range(n_clips):
        if target_mask[i]:
            ax.axvspan(i - 0.4, i + 0.4 + 2*width, alpha=0.1, color="red")

    # 1c: Success rate per clip
    ax = axes[1, 0]
    for i, name in enumerate(model_names):
        ax.bar(x + i * width, success_matrix[i], width, label=name, color=colors[i], alpha=0.85)
    ax.set_xlabel("Motion Clip")
    ax.set_ylabel("Success Rate")
    ax.set_title("(c) Per-Clip Success Rate")
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    for i in range(n_clips):
        if target_mask[i]:
            ax.axvspan(i - 0.4, i + 0.4 + 2*width, alpha=0.1, color="red")

    # 1d: Grouped bar: target vs non-target
    ax = axes[1, 1]
    target_tracking = tracking_matrix[:, list(TARGET_CLIP_INDICES)].mean(axis=1)
    nontarget_mask = [i for i in range(n_clips) if i not in TARGET_CLIP_INDICES]
    nontarget_tracking = tracking_matrix[:, nontarget_mask].mean(axis=1)

    bar_x = np.arange(n_models)
    bars1 = ax.bar(bar_x - 0.2, target_tracking, 0.35, label="Target clips (fight)", color="#F44336", alpha=0.85)
    bars2 = ax.bar(bar_x + 0.2, nontarget_tracking, 0.35, label="Non-target clips (mean)", color="#4CAF50", alpha=0.85)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Tracking Reward (normalized)")
    ax.set_title("(d) Target vs Non-target Performance")
    ax.set_xticks(bar_x)
    ax.set_xticklabels(["Base", "Model A\n(λ=8.0)", "Model B\n(λ=16.0)"], fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "exp1_unlearning_validity.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "exp1_unlearning_validity.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Radar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle("Per-Clip Tracking Reward Comparison", fontsize=14, fontweight="bold", y=0.98)

    angles = np.linspace(0, 2 * np.pi, n_clips, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    for i, name in enumerate(model_names):
        values = tracking_matrix[i].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=colors[i], markersize=4)
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # Highlight target clips on radar
    for idx in TARGET_CLIP_INDICES:
        angle = angles[idx]
        ax.axvline(x=angle, color="red", linestyle="--", alpha=0.4, linewidth=1)

    fig.savefig(OUTPUT_DIR / "exp1_radar_chart.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "exp1_radar_chart.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Figures saved: exp1_unlearning_validity.png/pdf, exp1_radar_chart.png/pdf")


if __name__ == "__main__":
    run_all_evaluations()
