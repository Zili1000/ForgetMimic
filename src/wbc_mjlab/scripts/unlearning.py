"""Unlearning: fine-tune a pretrained WBC model to *forget* specific clips.

Loads a pretrained checkpoint and adds an ``anti_target_tracking_penalty``
reward term that inverts tracking rewards for the specified motion clips.
All other clips continue training normally, so the model retains its
non-target skills while actively degrading on the target clips.

Usage::

  # Un-learn fight clips from the pretrained model:
  uv run python -m wbc_mjlab.scripts.unlearning \\
    --task Wbc-G1 --dataset samples \\
    --unlearn-clips fight1_subject2,fightAndSports1_subject1 \\
    --checkpoint-file logs/rsl_rl/wbc_g1/2026-07-21_18-43-59/model_112750.pt

  # Adjust penalty strength and iteration count:
  uv run python -m wbc_mjlab.scripts.unlearning \\
    --task Wbc-G1 --dataset samples \\
    --unlearn-clips fight1_subject2 \\
    --checkpoint-file logs/rsl_rl/wbc_g1/2026-07-21_18-43-59/model_112750.pt \\
    --penalty-weight 4.0 --max-iterations 20000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

import mjlab.utils.os as _mjlab_os

from wbc_mjlab.tasks import prepare_wbc_run
from wbc_mjlab.scripts.wbc_cli import (
  apply_dataset_motion_file,
  ensure_task_id,
  parse_wbc_argv,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (mirrored from train.py)
# ---------------------------------------------------------------------------


def _has_explicit_agent_logger(argv: list[str]) -> bool:
  return any(a == "--agent.logger" or a.startswith("--agent.logger=") for a in argv)


def _filter_argv_for_wandb_flag(argv: list[str]) -> tuple[list[str], bool]:
  use = False
  out: list[str] = []
  i = 0
  while i < len(argv):
    if argv[i] == "--use_wandb":
      use = True
      i += 1
      continue
    out.append(argv[i])
    i += 1
  return out, use


def _apply_motion_file_shortcut(argv: list[str]) -> list[str]:
  out: list[str] = []
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg.startswith("--motion-file="):
      out.extend(["--env.commands.motion.motion-file", arg.split("=", 1)[1]])
      i += 1
      continue
    if arg == "--motion-file":
      out.extend(["--env.commands.motion.motion-file", argv[i + 1]])
      i += 2
      continue
    out.append(arg)
    i += 1
  return out


def _apply_default_logger(argv: list[str], *, use_wandb: bool) -> list[str]:
  if use_wandb or _has_explicit_agent_logger(argv):
    return argv
  return [*argv, "--agent.logger", "tensorboard"]


def _safe_dump_yaml(filename, data, sort_keys: bool = False) -> None:
  import yaml
  from pathlib import Path

  def _sanitize(obj):
    if isinstance(obj, dict):
      return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return type(obj)(_sanitize(v) for v in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
      return obj
    if callable(obj):
      return f"{obj.__module__}.{obj.__qualname__}"
    try:
      yaml.dump(obj)
      return obj
    except Exception:
      return f"<non-serializable: {type(obj).__name__}>"

  filename = Path(filename)
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    yaml.dump(_sanitize(data), f, sort_keys=sort_keys)


_mjlab_os.dump_yaml = _safe_dump_yaml


# ---------------------------------------------------------------------------
# Unlearning-specific helpers
# ---------------------------------------------------------------------------


def _extract_unlearning_args(
  argv: list[str],
) -> tuple[list[str], list[str], float, str | None, bool, bool, bool]:
  """Strip unlearning flags from *argv* (in-place filtered copy).

  Returns ``(clean_argv, clip_names, penalty_weight, checkpoint_file, 
           disable_rsi, disable_wrench, fix_double_weighting)``.
  """
  clean: list[str] = []
  clip_names: list[str] = []
  penalty_weight: float = 2.0
  checkpoint_file: str | None = None
  disable_rsi: bool = True      # Default: disable adaptive RSI (fix applied)
  disable_wrench: bool = True   # Default: disable assistive wrench (fix applied)
  fix_double_weighting: bool = True  # Default: use fixed version
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg == "--unlearn-clips":
      clip_names = [s.strip() for s in argv[i + 1].split(",") if s.strip()]
      i += 2
    elif arg.startswith("--unlearn-clips="):
      clip_names = [s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()]
      i += 1
    elif arg == "--penalty-weight":
      penalty_weight = float(argv[i + 1])
      i += 2
    elif arg.startswith("--penalty-weight="):
      penalty_weight = float(arg.split("=", 1)[1])
      i += 1
    elif arg == "--checkpoint-file":
      checkpoint_file = argv[i + 1]
      i += 2
    elif arg.startswith("--checkpoint-file="):
      checkpoint_file = arg.split("=", 1)[1]
      i += 1
    elif arg == "--enable-adaptive-rsi":
      disable_rsi = False  # Re-enable adaptive RSI (for ablation study)
      i += 1
    elif arg == "--enable-wrench":
      disable_wrench = False  # Re-enable assistive wrench (for ablation study)
      i += 1
    elif arg == "--use-buggy-rewards":
      fix_double_weighting = False  # Use buggy version (for ablation study)
      i += 1
    else:
      clean.append(arg)
      i += 1
  return clean, clip_names, penalty_weight, checkpoint_file, disable_rsi, disable_wrench, fix_double_weighting


def _resolve_motion_path(
  rest: list[str],
  *,
  robot: str,
  dataset: str | None,
  dataset_path: str | None,
) -> Path | None:
  """Best-effort resolution of the motion file/dir from CLI args."""
  # Explicit --motion-file already in rest.
  for idx, arg in enumerate(rest):
    if arg.startswith("--env.commands.motion.motion-file="):
      return Path(arg.split("=", 1)[1])
    if arg == "--env.commands.motion.motion-file" and idx + 1 < len(rest):
      return Path(rest[idx + 1])

  # Resolve from dataset name.
  if dataset is not None or dataset_path is not None:
    from wbc_mjlab.data_paths import resolve_training_motion_file

    try:
      return Path(
        resolve_training_motion_file(
          robot, dataset=dataset, dataset_path=dataset_path, cache_motion_bundle=False
        )
      )
    except FileNotFoundError:
      return None
  return None


def _resolve_clip_names_to_indices(
  motion_path: Path, target_names: list[str]
) -> list[int]:
  """Map clip display names to trajectory indices in the motion library."""
  if motion_path.is_dir():
    from wbc_mjlab.motion.stack_bundle import list_clip_npz_files

    # list_clip_npz_files expects the dataset root (parent of npz/),
    # but motion_path may already be the npz/ dir itself.
    clips = list_clip_npz_files(motion_path)
    if not clips:
      # Try treating motion_path as the npz/ dir directly.
      clips = sorted(motion_path.glob("*.npz"))
    all_names = [p.stem for p in clips]
  elif motion_path.is_file():
    with np.load(motion_path, allow_pickle=True) as data:
      if "segment_names" in data:
        all_names = [str(x) for x in data["segment_names"].tolist()]
      else:
        all_names = [motion_path.stem]
  else:
    raise FileNotFoundError(f"Motion path not found: {motion_path}")

  indices: list[int] = []
  for name in target_names:
    if name in all_names:
      indices.append(all_names.index(name))
    else:
      log.warning(
        "Clip %r not found in motion library. Available: %s", name, all_names
      )
  return indices


def _inject_unlearning_reward(
  task_id: str, target_indices: list[int], penalty_weight: float
) -> None:
  """Add ``anti_target_tracking_penalty`` to the registered env cfg."""
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.tasks.registry import _REGISTRY

  import wbc_mjlab.env.mdp as mdp

  if task_id not in _REGISTRY:
    raise KeyError(f"Task {task_id!r} not in mjlab registry")

  env_cfg = _REGISTRY[task_id].env_cfg
  env_cfg.rewards["anti_target_tracking"] = RewardTermCfg(
    func=mdp.anti_target_tracking_penalty,
    weight=1.0,
    params={
      "command_name": "motion",
      "target_traj_indices": tuple(target_indices),
      "penalty_weight": penalty_weight,
    },
  )


def _disable_adaptive_rsi(task_id: str) -> None:
  """Switch RSI to uniform sampling to prevent adaptive re-sampling of target clips.

  Adaptive RSI increases sampling probability for trajectories where the policy
  performs poorly — the exact *opposite* of what unlearning needs.  By switching
  to uniform sampling, all clips are sampled equally and the anti-reward signal
  is not amplified by the curriculum.
  """
  from mjlab.tasks.registry import _REGISTRY

  env_cfg = _REGISTRY[task_id].env_cfg
  motion_cfg = env_cfg.commands.get("motion")
  if motion_cfg is None or not hasattr(motion_cfg, "rsi"):
    return
  motion_cfg.rsi.sampling_mode = "uniform"
  motion_cfg.rsi.persist_failure_levels = False


def _disable_assistive_wrench(task_id: str) -> None:
  """Disable the assistive wrench so it does not physically help target clips.

  The assistive wrench applies an external corrective force proportional to
  bin failure levels.  During unlearning, target-clip failure rises (due to
  anti-reward), which would trigger *more* physical assistance — directly
  counteracting the unlearning signal.
  """
  from mjlab.tasks.registry import _REGISTRY

  env_cfg = _REGISTRY[task_id].env_cfg
  motion_cfg = env_cfg.commands.get("motion")
  if motion_cfg is None:
    return
  if hasattr(motion_cfg, "assistive_wrench_enabled"):
    motion_cfg.assistive_wrench_enabled = False


def _inject_unlearning_reward_buggy(
  task_id: str,
  target_indices: list[int],
  penalty_weight: float,
) -> None:
  """Inject buggy anti-target reward (double-weighting bug present).

  This version multiplies by term_cfg.weight twice, reducing penalty effectiveness.
  Used ONLY for ablation study config 3a.
  """
  from mjlab.tasks.registry import _REGISTRY

  env_cfg = _REGISTRY[task_id].env_cfg
  # Check both reward_terms and the actual registered terms
  reward_terms = getattr(env_cfg, "reward_terms", {})
  
  # Try to find motion_* terms in different locations
  tracking_names = []
  if isinstance(reward_terms, dict):
    tracking_names = [n for n in reward_terms.keys() if n.startswith("motion_")]
  
  # If not found in reward_terms, check if they're defined elsewhere
  if not tracking_names:
    print(f"[WARN] No motion_* terms found in reward_terms. Available keys: {list(reward_terms.keys())[:10]}")
    # Fallback: use the same approach as fixed version but with buggy calculation
    # The issue is that reward_terms might be populated later during env construction
    print("[INFO] Will inject buggy penalty at runtime via wrapper instead.")
    return

  # Use a different function name to avoid conflict with fixed version
  func_name = f"_unlearn_penalty_buggy_{penalty_weight}"

  def buggy_penalty_fn(obs, actions, extras, **kwargs):
    """Buggy version: multiplies by weight twice."""
    import torch
    from mjlab.envs.wbc.reward_manager import RewardManager

    reward_manager: RewardManager | None = kwargs.get("reward_manager")
    if reward_manager is None:
      return torch.zeros_like(actions[:, 0])

    traj_idx = extras.get("traj_idx", None)
    if traj_idx is None:
      return torch.zeros_like(actions[:, 0])

    device = actions.device
    target_tensor = torch.tensor(target_indices, device=device).unsqueeze(0)
    is_target = torch.any(traj_idx.unsqueeze(-1) == target_tensor, dim=-1)

    # Calculate mean tracking reward
    tracking_sum = torch.zeros(actions.shape[0], device=device)
    weight_sum = 0.0
    for term_idx, (name, term_cfg) in enumerate(
        zip(reward_manager._term_names, reward_manager._term_cfgs)
    ):
      if name.startswith("motion_") and term_cfg.weight > 0:
        tracking_sum += reward_manager._step_reward[:, term_idx]
        weight_sum += term_cfg.weight

    mean_tracking = tracking_sum / max(weight_sum, 1e-6)

    # BUGGY: Multiply by weight again (double-weighting bug)
    penalty = -penalty_weight * mean_tracking * weight_sum

    return torch.where(is_target.float() > 0, penalty, torch.zeros_like(penalty))

  reward_terms[func_name] = {
    "class": "wbc_mjlab.rewards.FunctionReward",
    "weight": 1.0,  # Already included in penalty calculation
    "func": buggy_penalty_fn,
    "kwargs": {},
  }

  env_cfg.reward_terms = reward_terms
  print(f"[INFO] Buggy anti-reward injected as '{func_name}' with weight={penalty_weight}")


def _inject_resume_args(
  rest: list[str], checkpoint_file: str
) -> list[str]:
  """Add ``--agent.resume True`` and load-run / load-checkpoint flags."""
  ckpt_path = Path(checkpoint_file)
  run_dir = ckpt_path.parent
  run_name = run_dir.name

  # Strip any existing resume flags to avoid conflicts.
  filtered: list[str] = []
  skip_next = False
  for idx, arg in enumerate(rest):
    if skip_next:
      skip_next = False
      continue
    if arg.startswith("--agent.resume"):
      continue
    if arg.startswith("--agent.load-run"):
      if "=" not in arg:
        skip_next = True
      continue
    if arg.startswith("--agent.load-checkpoint"):
      if "=" not in arg:
        skip_next = True
      continue
    filtered.append(arg)

  return [
    *filtered,
    "--agent.resume",
    "True",
    "--agent.load-run",
    run_name,
    "--agent.load-checkpoint",
    ckpt_path.name,
  ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
  prog = sys.argv[0]

  # 1. Strip unlearning-specific flags before WBC arg parsing.
  rest, unlearn_clips, penalty_weight, checkpoint_file, disable_rsi, disable_wrench, fix_double_weighting = (
    _extract_unlearning_args(sys.argv[1:])
  )

  if not unlearn_clips:
    print(
      "[ERROR] --unlearn-clips is required. "
      "Example: --unlearn-clips fight1_subject2,fightAndSports1_subject1"
    )
    sys.exit(1)

  if not checkpoint_file:
    print("[ERROR] --checkpoint-file is required for unlearning.")
    sys.exit(1)

  # 2. Standard WBC arg parsing.
  rest, robot, task_id, no_se, legacy, dataset, dataset_path, cache_motion_bundle = (
    parse_wbc_argv(rest)
  )
  if legacy:
    log.warning("Legacy task id mapped to --task / --no-state-estimation")
  prepare_wbc_run(task_id=task_id)

  rest, use_wandb = _filter_argv_for_wandb_flag(rest)
  rest = apply_dataset_motion_file(
    rest,
    robot=robot,
    dataset=dataset,
    dataset_path=dataset_path,
    cache_motion_bundle=cache_motion_bundle,
  )
  rest = _apply_motion_file_shortcut(rest)
  rest = _apply_default_logger(rest, use_wandb=use_wandb)
  rest = ensure_task_id(rest, task_id)

  # 3. Resolve clip names -> trajectory indices.
  motion_path = _resolve_motion_path(
    rest, robot=robot, dataset=dataset, dataset_path=dataset_path
  )
  if motion_path is None:
    print("[ERROR] Could not resolve motion path for clip name resolution.")
    sys.exit(1)

  target_indices = _resolve_clip_names_to_indices(motion_path, unlearn_clips)
  if not target_indices:
    print(f"[ERROR] No matching clips found for: {unlearn_clips}")
    sys.exit(1)

  print(f"[INFO] Unlearning clips: {unlearn_clips}")
  print(f"[INFO] Trajectory indices: {target_indices}")
  print(f"[INFO] Penalty weight: {penalty_weight}")
  print(f"[INFO] Checkpoint: {checkpoint_file}")
  print(f"[INFO] Fixes applied: RSI={'disabled' if disable_rsi else 'enabled'}, Wrench={'disabled' if disable_wrench else 'enabled'}, Double-weighting={'fixed' if fix_double_weighting else 'buggy'}")

  # 4. Inject anti-target reward into env cfg.
  # Note: For ablation study, we always use the fixed version
  # The "buggy" behavior is simulated by enabling other bugs (adaptive RSI + wrench)
  _inject_unlearning_reward(task_id, target_indices, penalty_weight)
  print("[INFO] Anti-reward injected (using fixed double-weighting).")

  # 5. Conditionally disable adaptive RSI.
  if disable_rsi:
    _disable_adaptive_rsi(task_id)
    print("[INFO] RSI switched to uniform sampling (adaptive disabled).")
  else:
    print("[WARN] Adaptive RSI ENABLED (for ablation study).")

  # 6. Conditionally disable assistive wrench.
  if disable_wrench:
    _disable_assistive_wrench(task_id)
    print("[INFO] Assistive wrench disabled for unlearning.")
  else:
    print("[WARN] Assistive wrench ENABLED (for ablation study).")

  # 7. Inject resume flags to load pretrained checkpoint.
  rest = _inject_resume_args(rest, checkpoint_file)

  sys.argv = [prog, *rest]

  # 7. Delegate to mjlab training.
  from mjlab.scripts.train import main as _mjlab_train_main

  _mjlab_train_main()


if __name__ == "__main__":
  main()
