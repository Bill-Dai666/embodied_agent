"""
Bar plots (mean by condition) and OLS predicted-vs-actual plots for mediation scripts.
Titles use the outcome variable name (e.g. EMP_Mean, anthropomorphism_score).
"""
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import pandas as pd


def save_mediation_figures(
    df: pd.DataFrame,
    outcome_col: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    outcome_dir: str,
) -> None:
    """
    Save under outcome_dir:
      - bar_by_condition.png
      - ols_actual_vs_predicted.png
    """
    label = outcome_col

    # 1) Mean outcome by experimental condition (with SEM error bars)
    try:
        cond_group = (
            df.groupby("condition")[outcome_col]
            .agg(["mean", "std", "count"])
            .dropna()
        )
        if not cond_group.empty:
            cond_group = cond_group.copy()
            cond_group["sem"] = cond_group["std"] / cond_group["count"].apply(
                lambda n: math.sqrt(n) if n and n > 0 else float("nan")
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                cond_group.index,
                cond_group["mean"],
                yerr=cond_group["sem"],
                capsize=4,
                color="#4C72B0",
            )
            ax.set_ylabel(label)
            ax.set_xlabel("Condition")
            ax.set_title(f"{label}: mean by condition", fontsize=10)
            # Fixed y-axis to match scale: Godspeed anthropomorphism items are 1–5; EMP composite/items 1–7
            if outcome_col == "anthropomorphism_score":
                ax.set_ylim(0, 5)
                ax.set_yticks(list(range(0, 6)))
            else:
                ax.set_ylim(0, 7)
                ax.set_yticks(list(range(0, 8)))
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            path = os.path.join(outcome_dir, "bar_by_condition.png")
            plt.savefig(path, dpi=300)
            plt.close(fig)
            print(f"    plot: {path}")
    except Exception as e:
        print(f"    Warning: bar plot failed ({label}): {e}")

    # 2) Actual vs predicted (OLS) + identity line
    try:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
        min_val = min(float(y_true.min()), float(y_pred.min()))
        max_val = max(float(y_true.max()), float(y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{label}: actual vs. predicted (OLS)", fontsize=10)
        plt.tight_layout()
        path = os.path.join(outcome_dir, "ols_actual_vs_predicted.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        print(f"    plot: {path}")
    except Exception as e:
        print(f"    Warning: actual vs predicted plot failed ({label}): {e}")
