
"""
plot_metrics.py
===============
Drop this into your training directory alongside local_grpo.py.
It hooks into GRPOTrainer via a callback, collects metrics every
logging_steps, and saves a multi-panel matplotlib figure when
training ends (or when you call plot_metrics() manually).
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformers import TrainerCallback


class MetricsLogger(TrainerCallback):
    """Collects GRPO training metrics and plots them at the end."""

    def __init__(self, output_path: str = "grpo_metrics.png"):
        self.output_path = output_path
        self.steps   = []
        self.loss    = []
        self.reward  = []
        self.kl      = []
        self.lr      = []
        self.grad    = []
        # per-reward-function logs (keyed by function name)
        self.reward_components: dict[str, list] = {}

    # ------------------------------------------------------------------
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        step = state.global_step
        self.steps.append(step)

        self.loss.append(logs.get("loss", float("nan")))
        self.lr.append(logs.get("learning_rate", float("nan")))
        self.grad.append(logs.get("grad_norm", float("nan")))

        # TRL logs aggregate reward as "reward" or "rewards/mean"
        reward_val = logs.get("reward", logs.get("rewards/mean", float("nan")))
        self.reward.append(reward_val)

        # KL divergence — TRL key varies by version
        kl_val = logs.get(
            "kl", logs.get("policy/kl", logs.get("mean_kl", float("nan")))
        )
        self.kl.append(kl_val)

        # Per-reward-function scores  e.g. "rewards/format_reward"
        for key, val in logs.items():
            if key.startswith("rewards/") and key != "rewards/mean":
                name = key.split("/", 1)[1]
                self.reward_components.setdefault(name, [])
                # pad so every component list stays the same length as self.steps
                while len(self.reward_components[name]) < len(self.steps) - 1:
                    self.reward_components[name].append(float("nan"))
                self.reward_components[name].append(val)

        # pad short component lists
        for name, vals in self.reward_components.items():
            while len(vals) < len(self.steps):
                vals.append(float("nan"))

    # ------------------------------------------------------------------
    def on_train_end(self, args, state, control, **kwargs):
        self.plot_metrics()

    # ------------------------------------------------------------------
    def plot_metrics(self):
        n_components = len(self.reward_components)
        n_rows = 3 if n_components == 0 else 4

        fig, axes = plt.subplots(
            n_rows, 2,
            figsize=(14, n_rows * 3.5),
            constrained_layout=True,
        )
        fig.suptitle("GRPO Training Metrics — Synnapse Qwen3-4B", fontsize=13, y=1.01)

        s = self.steps

        # ── 1. Policy loss ────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(s, self.loss, color="#E24B4A", linewidth=1.4)
        ax.set_title("Policy loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(True, linestyle="--", alpha=0.4)

        # ── 2. Mean reward ────────────────────────────────────────────
        ax = axes[0, 1]
        ax.plot(s, self.reward, color="#7F77DD", linewidth=1.4)
        ax.set_title("Mean reward")
        ax.set_xlabel("step")
        ax.set_ylabel("reward")
        ax.grid(True, linestyle="--", alpha=0.4)

        # ── 3. KL divergence ──────────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(s, self.kl, color="#BA7517", linewidth=1.4)
        ax.set_title("KL divergence (policy vs reference)")
        ax.set_xlabel("step")
        ax.set_ylabel("KL")
        ax.grid(True, linestyle="--", alpha=0.4)

        # ── 4. Learning rate ──────────────────────────────────────────
        ax = axes[1, 1]
        ax.plot(s, self.lr, color="#534AB7", linewidth=1.4)
        ax.set_title("Learning rate schedule")
        ax.set_xlabel("step")
        ax.set_ylabel("LR")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
        ax.grid(True, linestyle="--", alpha=0.4)

        # ── 5. Gradient norm ──────────────────────────────────────────
        ax = axes[2, 0]
        ax.plot(s, self.grad, color="#1D9E75", linewidth=1.4)
        ax.axhline(0.3, color="#E24B4A", linestyle="--", linewidth=0.9, label="max_grad_norm")
        ax.set_title("Gradient norm")
        ax.set_xlabel("step")
        ax.set_ylabel("‖g‖")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        # ── 6. Reward components (stacked lines) ──────────────────────
        colors = ["#1D9E75", "#378ADD", "#D85A30", "#BA7517", "#534AB7"]
        ax = axes[2, 1]
        if n_components > 0:
            for i, (name, vals) in enumerate(self.reward_components.items()):
                ax.plot(s, vals, label=name, color=colors[i % len(colors)], linewidth=1.3)
            ax.set_title("Per-reward-function scores")
            ax.set_xlabel("step")
            ax.set_ylabel("reward")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
        else:
            ax.set_visible(False)

        # ── 7 & 8 (row 3, only if components exist) ───────────────────
        if n_components > 0:
            # Final reward breakdown — bar chart
            ax = axes[3, 0]
            names = list(self.reward_components.keys())
            finals = [self.reward_components[n][-1] for n in names]
            bars = ax.bar(names, finals, color=colors[: len(names)])
            ax.set_title("Final reward breakdown")
            ax.set_ylabel("reward at last step")
            ax.set_ylim(0, 1.05)
            for bar, val in zip(bars, finals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

            # Loss vs reward scatter
            ax = axes[3, 1]
            sc = ax.scatter(
                self.loss, self.reward,
                c=s, cmap="viridis", s=18, alpha=0.7
            )
            plt.colorbar(sc, ax=ax, label="step")
            ax.set_title("Loss vs reward trajectory")
            ax.set_xlabel("loss")
            ax.set_ylabel("reward")
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.savefig(self.output_path, dpi=150, bbox_inches="tight")
        print(f"Metrics plot saved to: {self.output_path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Usage — add MetricsLogger to your GRPOTrainer in local_grpo.py:
#
#   from plot_metrics import MetricsLogger
#
#   trainer = GRPOTrainer(
#       ...
#       callbacks=[HeartbeatCallback(), MetricsLogger("grpo_metrics.png")],
#   )
# ---------------------------------------------------------------------------
