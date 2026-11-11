"""
Part 2: First-Price Sealed-Bid Auction Simulation
=================================================
Bidder 1 deviates from equilibrium by bidding k*v1.
The other (n-1) bidders bid equilibrium: ((n-1)/n)*vi, vi ~ Uniform[0,1].
We simulate across k ∈ {0.1, 0.2, ..., 2.0} and n ∈ {2, 3, 4}.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import os

# Reproducibility
np.random.seed(42)

NUM_AUCTIONS = 100_000
N_VALUES = [2, 3, 4]
K_VALUES = np.round(np.arange(0.1, 2.01, 0.1), 2)  # 0.1, 0.2, ..., 2.0  (20 values)
PLOTS_DIR = "plots"


def simulate_first_price(n: int, k: float) -> tuple[float, float]:
    """
    Simulate NUM_AUCTIONS first-price auctions.

    Bidder 1: b1 = k * v1,  v1 ~ Uniform[0,1]
    Rivals  : bi = ((n-1)/n) * vi,  vi ~ Uniform[0,1]

    Returns
    -------
    avg_utility   : mean utility of bidder 1 across all auctions
    overpaid_frac : fraction of *winning* auctions where bidder 1 overpaid
                    (i.e. max rival bid < b1, so a lower bid would still win)
    """
    eq_factor = (n - 1) / n

    # Bidder 1
    v1 = np.random.uniform(0, 1, NUM_AUCTIONS)
    b1 = k * v1

    # Rivals: shape (NUM_AUCTIONS, n-1)
    rival_v = np.random.uniform(0, 1, (NUM_AUCTIONS, n - 1))
    rival_b = eq_factor * rival_v

    # Highest rival bid per auction
    max_rival = rival_b.max(axis=1)

    # Bidder 1 wins when b1 > max_rival
    wins = b1 > max_rival

    # Utility: v1*(1-k) if win, else 0
    utility = np.where(wins, v1 * (1 - k), 0.0)
    avg_utility = utility.mean()

    # Overpaid fraction among wins
    winning_auctions = wins.sum()
    if winning_auctions == 0:
        overpaid_frac = np.nan
    else:
        # Overpaid iff bidder 1 won but max_rival < b1 (always true when they win
        # by strict inequality, but we want to count cases where they could have
        # lowered their bid and still won — i.e. max_rival < b1)
        overpaid = (max_rival[wins] < b1[wins]).sum()
        overpaid_frac = overpaid / winning_auctions

    return avg_utility, overpaid_frac


def run_simulation() -> dict:
    """Run all (n, k) combinations and collect results."""
    results = {}
    for n in N_VALUES:
        print(f"  Simulating n={n} ...", flush=True)
        utilities = []
        overpaid_fracs = []
        for k in K_VALUES:
            avg_u, op_frac = simulate_first_price(n, k)
            utilities.append(avg_u)
            overpaid_fracs.append(op_frac)
        results[n] = {
            "utilities": np.array(utilities),
            "overpaid_fracs": np.array(overpaid_fracs),
        }
        print(
            f"    n={n}: eq k={(n-1)/n:.3f}, "
            f"utility at eq={results[n]['utilities'][round((n-1)/n*10)-1]:.5f}",
            flush=True,
        )
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_vlines(ax: plt.Axes, n: int) -> None:
    """Add vertical lines for k=1 and equilibrium k=(n-1)/n."""
    eq_k = (n - 1) / n
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2,
               label="k=1 (truthful)")
    ax.axvline(x=eq_k, color="red", linestyle="-.", linewidth=1.2,
               label=f"k={eq_k:.3f} (eq.)")
    ax.legend(fontsize=7, loc="upper right")


def plot_utility(results: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle("Part 2 — First-Price Auction: Expected Utility vs k", fontsize=13)

    for ax, n in zip(axes, N_VALUES):
        utils = results[n]["utilities"]
        ax.plot(K_VALUES, utils, "b-o", markersize=4, linewidth=1.5)
        _add_vlines(ax, n)
        # Mark maximum
        best_idx = np.argmax(utils)
        ax.scatter(K_VALUES[best_idx], utils[best_idx], color="green",
                   zorder=5, s=60, label=f"max at k={K_VALUES[best_idx]:.1f}")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_xlabel("k (bid scaling factor)", fontsize=9)
        ax.set_ylabel("Average Utility of Bidder 1", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_overpaid(results: dict, path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle("Part 2 — First-Price Auction: Fraction Bidder 1 Overpaid (given win)", fontsize=12)

    for ax, n in zip(axes, N_VALUES):
        op = results[n]["overpaid_fracs"]
        ax.plot(K_VALUES, op, "m-s", markersize=4, linewidth=1.5)
        _add_vlines(ax, n)
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_xlabel("k (bid scaling factor)", fontsize=9)
        ax.set_ylabel("Fraction Overpaid (among wins)", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined(results: dict, path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Part 2 — First-Price Auction: Combined Summary", fontsize=13)

    for col_idx, n in enumerate(N_VALUES):
        utils = results[n]["utilities"]
        op = results[n]["overpaid_fracs"]

        # --- Top row: utility ---
        ax_u = axes[0, col_idx]
        ax_u.plot(K_VALUES, utils, "b-o", markersize=4, linewidth=1.5)
        _add_vlines(ax_u, n)
        best_idx = np.argmax(utils)
        ax_u.scatter(K_VALUES[best_idx], utils[best_idx], color="green",
                     zorder=5, s=60, label=f"max k={K_VALUES[best_idx]:.1f}")
        ax_u.legend(fontsize=7)
        ax_u.set_title(f"n={n} — Utility", fontsize=10)
        ax_u.set_xlabel("k", fontsize=9)
        ax_u.set_ylabel("Avg Utility", fontsize=9)
        ax_u.grid(True, alpha=0.3)

        # --- Bottom row: overpaid fraction ---
        ax_o = axes[1, col_idx]
        ax_o.plot(K_VALUES, op, "m-s", markersize=4, linewidth=1.5)
        _add_vlines(ax_o, n)
        ax_o.set_title(f"n={n} — Overpaid Fraction", fontsize=10)
        ax_o.set_xlabel("k", fontsize=9)
        ax_o.set_ylabel("Frac. Overpaid", fontsize=9)
        ax_o.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY: First-Price Auction Results")
    print("=" * 60)
    for n in N_VALUES:
        eq_k = (n - 1) / n
        eq_idx = int(round(eq_k * 10)) - 1  # index in K_VALUES for eq k
        utils = results[n]["utilities"]
        op = results[n]["overpaid_fracs"]
        best_idx = np.argmax(utils)
        print(f"\nn = {n}  |  Equilibrium k = {eq_k:.4f}")
        print(f"  k that maximises simulated utility : {K_VALUES[best_idx]:.2f}")
        print(f"  Utility at equilibrium k           : {utils[eq_idx]:.6f}")
        print(f"  Utility at best simulated k        : {utils[best_idx]:.6f}")
        print(f"  Overpaid fraction at equilibrium k : {op[eq_idx]:.4f}")
        print(f"  Overpaid fraction at k=1 (truthful): {op[K_VALUES.tolist().index(1.0)]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Running Part 2: First-Price Auction Simulation")
    print(f"  Auctions per combination : {NUM_AUCTIONS:,}")
    print(f"  k grid                   : {K_VALUES.tolist()}")
    print(f"  n values                 : {N_VALUES}\n")

    results = run_simulation()

    print("\nGenerating plots ...")
    plot_utility(results, os.path.join(PLOTS_DIR, "part2_first_price_utility.png"))
    plot_overpaid(results, os.path.join(PLOTS_DIR, "part2_first_price_overpaid.png"))
    plot_combined(results, os.path.join(PLOTS_DIR, "part2_combined.png"))

    print_summary(results)
    print("\nPart 2 complete.")


if __name__ == "__main__":
    main()
