"""
Part 1: Second-Price Sealed-Bid Auction Simulation
===================================================
Simulates second-price (Vickrey) auctions to analyze bidding strategies.

Parts:
  A - One rival bidder with three strategies
  B - Fixed rival bid (exact computation, no simulation)
  C - Two rival bidders with three strategies
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# Constants
N_SIM = 100_000                      # auctions per combination
V1_VALUES = [0.2, 0.5, 0.8]         # bidder 1's private values
BIDS = np.linspace(0, 1, 101)       # 0.00, 0.01, ..., 1.00
RIVAL_STRATEGIES = ["Truthful (b2=v2)", "Half-value (b2=v2/2)", "Random Uniform[0,1]"]
FIXED_RIVAL_BIDS = [0.1, 0.4, 0.9]

PLOTS_DIR = "plots"


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Part A — One rival bidder
# ---------------------------------------------------------------------------

def simulate_one_rival(v1: float, strategy_idx: int) -> np.ndarray:
    """Return array of average utilities for each bid in BIDS (101 values).

    strategy_idx:
      0 — truthful   b2 = v2
      1 — half-value b2 = v2/2
      2 — random     b2 ~ Uniform[0,1]
    """
    v2 = np.random.uniform(0, 1, N_SIM)   # rival private values

    if strategy_idx == 0:
        b2 = v2
    elif strategy_idx == 1:
        b2 = v2 / 2.0
    else:
        b2 = np.random.uniform(0, 1, N_SIM)

    avg_utilities = np.empty(len(BIDS))
    for j, b1 in enumerate(BIDS):
        win_mask = b1 > b2
        utility = np.where(win_mask, v1 - b2, 0.0)
        avg_utilities[j] = utility.mean()

    return avg_utilities


def part_a():
    ensure_plots_dir()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    fig.suptitle("Part A: Second-Price Auction — One Rival Bidder\nAverage Utility of Bidder 1 vs Bid Amount",
                 fontsize=14, fontweight="bold")

    for row, v1 in enumerate(V1_VALUES):
        for col, strat_name in enumerate(RIVAL_STRATEGIES):
            ax = axes[row][col]
            avg_utils = simulate_one_rival(v1, col)
            opt_idx = np.argmax(avg_utils)
            opt_bid = BIDS[opt_idx]

            ax.plot(BIDS, avg_utils, color="steelblue", linewidth=1.8)
            ax.axvline(opt_bid, color="crimson", linestyle="--", linewidth=1.5,
                       label=f"Optimal b1={opt_bid:.2f}")
            ax.axvline(v1, color="green", linestyle=":", linewidth=1.2,
                       label=f"Truthful b1={v1:.1f}")

            ax.set_title(f"v1={v1}, {strat_name}", fontsize=9)
            ax.set_xlabel("Bid b1", fontsize=8)
            ax.set_ylabel("Avg Utility", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, "part1A_one_rival.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Part A] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Part B — Fixed rival bid (exact computation)
# ---------------------------------------------------------------------------

def compute_fixed_rival_utility(v1: float, fixed_b2: float) -> np.ndarray:
    """Exact expected utility: if b1 > fixed_b2, utility = v1 - fixed_b2; else 0."""
    return np.where(BIDS > fixed_b2, v1 - fixed_b2, 0.0)


def part_b():
    ensure_plots_dir()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    fig.suptitle("Part B: Second-Price Auction — Fixed Rival Bid (Exact)\n"
                 "Utility of Bidder 1 vs Bid Amount (truthful bid is weakly dominant)",
                 fontsize=14, fontweight="bold")

    for row, v1 in enumerate(V1_VALUES):
        for col, fb2 in enumerate(FIXED_RIVAL_BIDS):
            ax = axes[row][col]
            utils = compute_fixed_rival_utility(v1, fb2)

            ax.plot(BIDS, utils, color="steelblue", linewidth=1.8)

            # Mark truthful bid (b1 = v1) — weakly dominant strategy
            ax.axvline(v1, color="green", linestyle="--", linewidth=2.0,
                       label=f"Truthful b1={v1:.1f} (weakly dominant)")

            # Annotate the step: utility jumps at b1 = fixed_b2
            ax.axvline(fb2, color="orange", linestyle=":", linewidth=1.4,
                       label=f"Rival bid={fb2}")

            if v1 > fb2:
                ax.set_title(f"v1={v1}, fixed b2={fb2}\n→ b1>b2 wins; any b1∈({fb2:.1f},1] optimal", fontsize=8)
            else:
                ax.set_title(f"v1={v1}, fixed b2={fb2}\n→ v1<b2: winning gives negative utility", fontsize=8)

            ax.set_xlabel("Bid b1", fontsize=8)
            ax.set_ylabel("Utility", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, "part1B_fixed_rival.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Part B] Saved: {out_path}")

    # Verify truthfulness as weakly dominant
    print("\n  [Part B] Truthful dominance analysis:")
    for v1 in V1_VALUES:
        for fb2 in FIXED_RIVAL_BIDS:
            utils = compute_fixed_rival_utility(v1, fb2)
            # Utility at truthful bid
            truthful_idx = np.argmin(np.abs(BIDS - v1))
            u_truthful = utils[truthful_idx]
            # Truthful is weakly dominant: no bid gives strictly higher utility
            best_other = np.max(utils[np.abs(BIDS - v1) > 0.015])
            is_weakly_dominant = u_truthful >= best_other - 1e-9
            print(f"    v1={v1}, fixed_b2={fb2}: "
                  f"u(truthful={v1:.2f})={u_truthful:.4f}, "
                  f"best_other={best_other:.4f}, "
                  f"truthful_weakly_dominant={is_weakly_dominant}")


# ---------------------------------------------------------------------------
# Part C — Two rival bidders
# ---------------------------------------------------------------------------

def simulate_two_rivals(v1: float, strategy_idx: int) -> np.ndarray:
    """Return array of average utilities for each bid in BIDS with 2 rivals.

    Both rivals use the same strategy. Bidder 1 wins if b1 > max(b2, b3).
    Payment = max(b2, b3).
    """
    v2 = np.random.uniform(0, 1, N_SIM)
    v3 = np.random.uniform(0, 1, N_SIM)

    if strategy_idx == 0:
        b2, b3 = v2, v3
    elif strategy_idx == 1:
        b2, b3 = v2 / 2.0, v3 / 2.0
    else:
        b2 = np.random.uniform(0, 1, N_SIM)
        b3 = np.random.uniform(0, 1, N_SIM)

    second_price = np.maximum(b2, b3)   # payment if bidder 1 wins

    avg_utilities = np.empty(len(BIDS))
    for j, b1 in enumerate(BIDS):
        win_mask = b1 > second_price
        utility = np.where(win_mask, v1 - second_price, 0.0)
        avg_utilities[j] = utility.mean()

    return avg_utilities


def part_c():
    ensure_plots_dir()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    fig.suptitle("Part C: Second-Price Auction — Two Rival Bidders\nAverage Utility of Bidder 1 vs Bid Amount",
                 fontsize=14, fontweight="bold")

    for row, v1 in enumerate(V1_VALUES):
        for col, strat_name in enumerate(RIVAL_STRATEGIES):
            ax = axes[row][col]
            avg_utils = simulate_two_rivals(v1, col)
            opt_idx = np.argmax(avg_utils)
            opt_bid = BIDS[opt_idx]

            ax.plot(BIDS, avg_utils, color="darkorange", linewidth=1.8)
            ax.axvline(opt_bid, color="crimson", linestyle="--", linewidth=1.5,
                       label=f"Optimal b1={opt_bid:.2f}")
            ax.axvline(v1, color="green", linestyle=":", linewidth=1.2,
                       label=f"Truthful b1={v1:.1f}")

            ax.set_title(f"v1={v1}, {strat_name}", fontsize=9)
            ax.set_xlabel("Bid b1", fontsize=8)
            ax.set_ylabel("Avg Utility", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, "part1C_two_rivals.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [Part C] Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Part 1: Second-Price Sealed-Bid Auction Simulation")
    print("=" * 60)

    print("\n[Running Part A — One Rival Bidder]")
    part_a()
    print("  Part A complete.")

    print("\n[Running Part B — Fixed Rival Bid (Exact Computation)]")
    part_b()
    print("  Part B complete.")

    print("\n[Running Part C — Two Rival Bidders]")
    part_c()
    print("  Part C complete.")

    print("\nAll parts finished. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
