"""
Part 1: Second-Price Sealed-Bid (Vickrey) Auction Simulation
============================================================
Autonomous Systems - Assignment 4
Authors: Akanksha Singh (B23CM1005), Shweta Mandal (B23CM1037)

Simulates second-price sealed-bid auctions to evaluate the weakly dominant
nature of truthful bidding under diverse rival strategies and valuations.

Sections:
  A - One rival bidder with three strategies (Truthful, Half-Value, Random Uniform)
  B - Fixed rival bid analytical verification (b2 in {0.1, 0.4, 0.9})
  C - Two rival bidders (Bidder 2 and Bidder 3) with three strategies
"""

import os
import tempfile

# Prevent matplotlib cache permission warning
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Global Simulation Constants
N_SIM = 100_000                      # Auctions simulated per bid
V1_VALUES = [0.2, 0.5, 0.8]         # Bidder 1 private valuations
BIDS = np.linspace(0.0, 1.0, 101)   # Grid: 0.00, 0.01, ..., 1.00
RIVAL_STRATEGIES = [
    "Truthful ($b_2 = v_2$)",
    "Half-value ($b_2 = v_2 / 2$)",
    "Random Uniform ($b_2 \\sim U[0, 1]$)"
]
FIXED_RIVAL_BIDS = [0.1, 0.4, 0.9]
PLOTS_DIR = "plots"


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Part A: One Rival Bidder with 3 Strategies
# ---------------------------------------------------------------------------

def simulate_one_rival(v1: float, strategy_idx: int) -> np.ndarray:
    """
    Simulate auctions for Bidder 1 vs One Rival Bidder across all BIDS.

    Parameters:
      v1: True private value of Bidder 1.
      strategy_idx: 0 -> Truthful, 1 -> Half-value, 2 -> Random Uniform.

    Returns:
      Array of average utilities for each bid in BIDS (101 values).
    """
    v2 = np.random.uniform(0.0, 1.0, N_SIM)

    if strategy_idx == 0:
        b2 = v2
    elif strategy_idx == 1:
        b2 = v2 / 2.0
    else:
        b2 = np.random.uniform(0.0, 1.0, N_SIM)

    avg_utilities = np.empty(len(BIDS))
    for j, b1 in enumerate(BIDS):
        win = b1 > b2
        tie = b1 == b2
        utility = np.where(win, v1 - b2, np.where(tie, 0.5 * (v1 - b2), 0.0))
        avg_utilities[j] = utility.mean()

    return avg_utilities


def part_a():
    """Execute Part A and plot the 3x3 grid of average utilities."""
    ensure_plots_dir()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    fig.suptitle(
        "Part 1A: Second-Price Auction — One Rival Bidder\n"
        "Average Utility of Bidder 1 vs Bid Amount ($N = 100,000$ auctions per point)",
        fontsize=14, fontweight="bold", y=0.99
    )

    print("\n--- Part 1A: One Rival Bidder Analysis ---")
    for row, v1 in enumerate(V1_VALUES):
        for col, strat_name in enumerate(RIVAL_STRATEGIES):
            ax = axes[row][col]
            avg_utils = simulate_one_rival(v1, col)

            max_util = np.max(avg_utils)
            optimal_mask = avg_utils >= (max_util - 1e-4)
            optimal_bids = BIDS[optimal_mask]
            min_opt, max_opt = optimal_bids[0], optimal_bids[-1]

            opt_idx = np.argmax(avg_utils)
            opt_bid = BIDS[opt_idx]

            truthful_idx = int(round(v1 * 100))
            u_truthful = avg_utils[truthful_idx]

            ax.plot(BIDS, avg_utils, color="#1f77b4", linewidth=2.0, label="Avg Utility")
            ax.axvline(v1, color="forestgreen", linestyle="--", linewidth=1.8,
                       label=f"Truthful $b_1={v1:.1f}$ (u={u_truthful:.4f})")

            if max_opt - min_opt > 0.05:
                ax.axvspan(min_opt, max_opt, color="lightgreen", alpha=0.25,
                           label=f"Max Plateau [{min_opt:.2f}, {max_opt:.2f}]")
            else:
                ax.axvline(opt_bid, color="crimson", linestyle=":", linewidth=1.5,
                           label=f"Sim Peak $b_1={opt_bid:.2f}$")

            ax.set_title(f"Bidder 1 Value $v_1={v1}$ | Rival: {strat_name}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Bidder 1 Bid ($b_1$)", fontsize=9)
            ax.set_ylabel("Average Utility", fontsize=9)
            ax.legend(fontsize=7.5, loc="best")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=8)

            print(f"  v1={v1:.1f} | Strategy {col} ({strat_name[:12]}): "
                  f"Truthful u={u_truthful:.4f}, Peak u={max_util:.4f} at b1={opt_bid:.2f} "
                  f"(optimal range: [{min_opt:.2f}, {max_opt:.2f}])")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PLOTS_DIR, "part1a_one_rival.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Part B: Fixed Rival Bid Analytical Verification
# ---------------------------------------------------------------------------

def compute_fixed_rival_utility(v1: float, fixed_b2: float) -> np.ndarray:
    """Exact analytical utility of Bidder 1 against a fixed rival bid b2."""
    win = BIDS > fixed_b2
    tie = np.isclose(BIDS, fixed_b2, atol=1e-5)
    return np.where(win, v1 - fixed_b2, np.where(tie, 0.5 * (v1 - fixed_b2), 0.0))


def part_b():
    """Execute Part B: verify b1 = v1 gives highest utility in all fixed cases."""
    ensure_plots_dir()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    fig.suptitle(
        "Part 1B: Second-Price Auction — Fixed Rival Bid Analytical Verification\n"
        "Utility of Bidder 1 vs Bid Amount ($b_1 = v_1$ is weakly dominant)",
        fontsize=14, fontweight="bold", y=0.99
    )

    print("\n--- Part 1B: Fixed Rival Bid Dominance Verification ---")
    for row, v1 in enumerate(V1_VALUES):
        for col, fb2 in enumerate(FIXED_RIVAL_BIDS):
            ax = axes[row][col]
            utils = compute_fixed_rival_utility(v1, fb2)

            truthful_idx = int(round(v1 * 100))
            u_truthful = utils[truthful_idx]
            max_util = np.max(utils)
            is_dominant = np.isclose(u_truthful, max_util, atol=1e-9)

            ax.plot(BIDS, utils, color="#1f77b4", linewidth=2.2, label="Exact Utility $u(b_1)$")
            ax.axvline(v1, color="forestgreen", linestyle="--", linewidth=2.0,
                       label=f"Truthful $b_1={v1:.1f}$ ($u={u_truthful:.2f}$)")
            ax.axvline(fb2, color="darkorange", linestyle=":", linewidth=1.8,
                       label=f"Rival bid $b_2={fb2:.1f}$")

            if v1 > fb2:
                status_txt = f"$v_1 > b_2$: Winning is profitable\nMax utility = {v1-fb2:.2f} for any $b_1 > {fb2:.1f}$"
            elif v1 < fb2:
                status_txt = f"$v_1 < b_2$: Winning causes loss ({v1-fb2:.2f})\nMax utility = 0.00 for any $b_1 < {fb2:.1f}$"
            else:
                status_txt = f"$v_1 = b_2$: Breakeven; utility is 0.00 for all bids"

            ax.set_title(f"$v_1={v1:.1f}$, Fixed $b_2={fb2:.1f}$\n{status_txt}", fontsize=9)
            ax.set_xlabel("Bidder 1 Bid ($b_1$)", fontsize=9)
            ax.set_ylabel("Utility", fontsize=9)
            ax.legend(fontsize=7.5, loc="best")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=8)

            print(f"  v1={v1:.1f}, fixed_b2={fb2:.1f} -> "
                  f"u(truthful)={u_truthful:.4f}, max_u={max_util:.4f}, "
                  f"Truthful is optimal: {is_dominant}")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, "part1b_fixed_rival.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Part C: Two Rival Bidders (Bidder 2 and Bidder 3)
# ---------------------------------------------------------------------------

def simulate_two_rivals(v1: float, strategy_idx: int) -> np.ndarray:
    """
    Simulate auctions with Bidder 1 vs Two Rival Bidders.
    Bidder 1 wins iff b1 > max(b2, b3), paying max(b2, b3).
    """
    v2 = np.random.uniform(0.0, 1.0, N_SIM)
    v3 = np.random.uniform(0.0, 1.0, N_SIM)

    if strategy_idx == 0:
        b2, b3 = v2, v3
    elif strategy_idx == 1:
        b2, b3 = v2 / 2.0, v3 / 2.0
    else:
        b2 = np.random.uniform(0.0, 1.0, N_SIM)
        b3 = np.random.uniform(0.0, 1.0, N_SIM)

    second_price = np.maximum(b2, b3)

    avg_utilities = np.empty(len(BIDS))
    for j, b1 in enumerate(BIDS):
        win = b1 > second_price
        tie = b1 == second_price
        utility = np.where(win, v1 - second_price, np.where(tie, 0.5 * (v1 - second_price), 0.0))
        avg_utilities[j] = utility.mean()

    return avg_utilities


def part_c():
    """Execute Part C: Two rival bidders under stochastic strategies."""
    ensure_plots_dir()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    fig.suptitle(
        "Part 1C: Second-Price Auction — Two Rival Bidders (Bidders 2 & 3)\n"
        "Average Utility of Bidder 1 vs Bid Amount ($N = 100,000$ auctions per point)",
        fontsize=14, fontweight="bold", y=0.99
    )

    print("\n--- Part 1C: Two Rival Bidders Analysis ---")
    for row, v1 in enumerate(V1_VALUES):
        for col, strat_name in enumerate(RIVAL_STRATEGIES):
            ax = axes[row][col]
            avg_utils = simulate_two_rivals(v1, col)

            max_util = np.max(avg_utils)
            optimal_mask = avg_utils >= (max_util - 1e-4)
            optimal_bids = BIDS[optimal_mask]
            min_opt, max_opt = optimal_bids[0], optimal_bids[-1]

            opt_idx = np.argmax(avg_utils)
            opt_bid = BIDS[opt_idx]

            truthful_idx = int(round(v1 * 100))
            u_truthful = avg_utils[truthful_idx]

            ax.plot(BIDS, avg_utils, color="#d95f02", linewidth=2.0, label="Avg Utility (2 Rivals)")
            ax.axvline(v1, color="forestgreen", linestyle="--", linewidth=1.8,
                       label=f"Truthful $b_1={v1:.1f}$ (u={u_truthful:.4f})")

            if max_opt - min_opt > 0.05:
                ax.axvspan(min_opt, max_opt, color="lightgreen", alpha=0.25,
                           label=f"Max Plateau [{min_opt:.2f}, {max_opt:.2f}]")
            else:
                ax.axvline(opt_bid, color="crimson", linestyle=":", linewidth=1.5,
                           label=f"Sim Peak $b_1={opt_bid:.2f}$")

            ax.set_title(f"Bidder 1 Value $v_1={v1}$ | 2 Rivals: {strat_name}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Bidder 1 Bid ($b_1$)", fontsize=9)
            ax.set_ylabel("Average Utility", fontsize=9)
            ax.legend(fontsize=7.5, loc="best")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=8)

            print(f"  v1={v1:.1f} | Strategy {col} ({strat_name[:12]}): "
                  f"Truthful u={u_truthful:.4f}, Peak u={max_util:.4f} at b1={opt_bid:.2f} "
                  f"(optimal range: [{min_opt:.2f}, {max_opt:.2f}])")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PLOTS_DIR, "part1c_two_rivals.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Part 1: Second-Price Sealed-Bid Auction Simulation & Verification")
    print("=" * 70)

    print("\n[Running Part 1A — One Rival Bidder (100,000 auctions / point)]")
    part_a()

    print("\n[Running Part 1B — Fixed Rival Bid Analytical Verification]")
    part_b()

    print("\n[Running Part 1C — Two Rival Bidders (100,000 auctions / point)]")
    part_c()

    print("\n" + "=" * 70)
    print("Part 1 complete. Plots saved:")
    print("  - plots/part1a_one_rival.png")
    print("  - plots/part1b_fixed_rival.png")
    print("  - plots/part1c_two_rivals.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
