"""
Part 2: First-Price Sealed-Bid Auction Simulation & Equilibrium Deviation
=========================================================================
Autonomous Systems - Assignment 4
Authors: Akanksha Singh (B23CM1005), Shweta Mandal (B23CM1037)

Evaluates first-price sealed-bid auctions with n bidders (n in {2, 3, 4}).
Bidder 1 deviates from equilibrium by bidding b1 = k * v1, where k in [0.1, 2.0].
The remaining (n - 1) rival bidders adhere to the symmetric risk-neutral
Bayesian Nash Equilibrium (BNE) strategy: bi = ((n - 1) / n) * vi.

Key Metrics Evaluated:
  1. Expected utility of Bidder 1 vs deviation factor k (empirical vs analytical).
  2. Verification that k* = (n - 1) / n maximises expected utility.
  3. Fraction of auctions where the winner could have paid less and still won:
     NOTE ON OVERPAYMENT:
     In auction theory, 'overpayment' refers strictly to ex-post regret—the winner
     paying more than the minimum necessary bid (i.e. b_(1) > b_(2))—leaving
     money on the table. It does NOT mean the winner makes a financial loss.
     At equilibrium k* = (n-1)/n < 1, the winner always earns a strictly positive
     surplus u = v1 - b1 = v1*(1 - k*) > 0 for any v1 > 0.
  4. Average overpayment margin (winner's regret: b_(1) - b_(2)).
"""

import os
import tempfile

# Prevent matplotlib cache warning
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

NUM_AUCTIONS = 100_000
N_VALUES = [2, 3, 4]
BASE_GRID = np.round(np.arange(0.1, 2.01, 0.05), 2)
PLOTS_DIR = "plots"


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def theoretical_utility(k: float, n: int) -> float:
    """Closed-form theoretical expected utility of Bidder 1 bidding k*v1."""
    eq_k = (n - 1) / n
    if k >= 1.0:
        return (1.0 - k) * (1.0 / (n + 1))
    if k <= eq_k:
        return ((1.0 - k) / (n + 1.0)) * ((n * k / (n - 1.0)) ** (n - 1))
    else:
        v_grid = np.linspace(0.0, 1.0, 10_000)
        win_prob = np.minimum(1.0, (n * k * v_grid / (n - 1.0)) ** (n - 1))
        integrand = v_grid * (1.0 - k) * win_prob
        return float(np.trapezoid(integrand, v_grid))


def simulate_first_price(n: int, k: float) -> dict:
    """
    Simulate NUM_AUCTIONS auctions for given n and deviation factor k.
    Measures expected utility, win probability, and overpayment metrics.
    """
    eq_factor = (n - 1) / n

    v1 = np.random.uniform(0.0, 1.0, NUM_AUCTIONS)
    b1 = k * v1

    rival_v = np.random.uniform(0.0, 1.0, (NUM_AUCTIONS, n - 1))
    rival_b = eq_factor * rival_v

    max_rival = rival_b.max(axis=1)

    win_strict = b1 > max_rival
    tie = np.isclose(b1, max_rival, atol=1e-6)
    win_mask = win_strict | (tie & (np.random.uniform(0.0, 1.0, NUM_AUCTIONS) < 0.5))

    utility = np.where(win_mask, v1 * (1.0 - k), 0.0)
    avg_utility = float(utility.mean())
    win_prob = float(win_mask.mean())

    num_wins = int(win_mask.sum())
    if num_wins > 0:
        # Overpaid if b1 > max_rival (could have reduced bid to max_rival + epsilon)
        overpaid_b1 = (b1[win_mask] > max_rival[win_mask]).sum()
        frac_b1_overpaid = float(overpaid_b1 / num_wins)
        regret_b1 = float((b1[win_mask] - max_rival[win_mask]).mean())
    else:
        frac_b1_overpaid = np.nan
        regret_b1 = 0.0

    all_bids = np.column_stack([b1, rival_b])
    sorted_all_bids = np.sort(all_bids, axis=1)
    highest_bid = sorted_all_bids[:, -1]
    second_highest_bid = sorted_all_bids[:, -2]

    # Fraction where winner paid strictly more than second-highest bid
    overall_could_pay_less = (highest_bid > second_highest_bid).mean()
    overall_regret = float((highest_bid - second_highest_bid).mean())

    return {
        "avg_utility": avg_utility,
        "win_prob": win_prob,
        "frac_b1_overpaid": frac_b1_overpaid,
        "overall_frac_overpaid": float(overall_could_pay_less),
        "regret_b1": regret_b1,
        "overall_regret": overall_regret,
    }


def run_experiment():
    """Run full simulation suite across all n in {2, 3, 4}."""
    ensure_plots_dir()
    results = {}

    for n in N_VALUES:
        eq_k = (n - 1) / n
        k_grid = np.unique(np.sort(np.concatenate([BASE_GRID, [eq_k]])))
        print(f"\n[Simulating n = {n} Bidders] (Theoretical BNE k* = {eq_k:.4f})")

        u_list, win_list, op_b1_list, op_all_list = [], [], [], []
        regret_b1_list, regret_all_list, theo_u_list = [], [], []

        for k in k_grid:
            metrics = simulate_first_price(n, k)
            theo_u = theoretical_utility(k, n)

            u_list.append(metrics["avg_utility"])
            win_list.append(metrics["win_prob"])
            op_b1_list.append(metrics["frac_b1_overpaid"])
            op_all_list.append(metrics["overall_frac_overpaid"])
            regret_b1_list.append(metrics["regret_b1"])
            regret_all_list.append(metrics["overall_regret"])
            theo_u_list.append(theo_u)

        results[n] = {
            "k_grid": k_grid,
            "utilities": np.array(u_list),
            "theoretical_u": np.array(theo_u_list),
            "win_probs": np.array(win_list),
            "op_b1": np.array(op_b1_list),
            "op_all": np.array(op_all_list),
            "regret_b1": np.array(regret_b1_list),
            "regret_all": np.array(regret_all_list),
            "eq_k": eq_k,
        }

        best_idx = np.argmax(results[n]["utilities"])
        best_k = k_grid[best_idx]
        best_u = results[n]["utilities"][best_idx]
        eq_idx = np.argmin(np.abs(k_grid - eq_k))
        eq_u = results[n]["utilities"][eq_idx]

        print(f"  n={n}: Simulated peak at k = {best_k:.4f} (u = {best_u:.5f}) | "
              f"BNE k = {eq_k:.4f} (u = {eq_u:.5f})")
        print(f"  Fraction of auctions winner could have paid less: {results[n]['op_all'][eq_idx]*100:.2f}%")
        print(f"  Average overpayment margin at BNE (b_(1) - b_(2)): {results[n]['regret_all'][eq_idx]:.4f}")

    return results


# ---------------------------------------------------------------------------
# Plotting Routines
# ---------------------------------------------------------------------------

def plot_utility(results: dict):
    """Plot Expected Utility vs k for n in {2, 3, 4}."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    fig.suptitle("Part 2: First-Price Auction — Expected Utility of Bidder 1 vs Scaling Factor $k$\n"
                 "Bidder 1 bids $b_1 = k \\cdot v_1$; Rivals bid equilibrium $b_i = \\frac{n-1}{n} v_i$",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, n in zip(axes, N_VALUES):
        k_vals = results[n]["k_grid"]
        u_sim = results[n]["utilities"]
        u_theo = results[n]["theoretical_u"]
        eq_k = results[n]["eq_k"]

        best_idx = np.argmax(u_sim)
        best_k = k_vals[best_idx]
        best_u = u_sim[best_idx]

        ax.plot(k_vals, u_sim, "o-", color="#1f77b4", markersize=3.5, linewidth=1.8, label="Simulated Utility")
        ax.plot(k_vals, u_theo, "--", color="#2ca02c", linewidth=1.5, label="Theoretical BNE Curve")

        ax.axvline(eq_k, color="crimson", linestyle="-.", linewidth=1.8,
                   label=f"BNE $k^*={eq_k:.3f}$ ($u={theoretical_utility(eq_k, n):.4f}$)")
        ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Truthful $k=1.0$ ($u=0$)")
        ax.scatter([best_k], [best_u], color="darkorange", s=70, zorder=5,
                   label=f"Sim Peak $k={best_k:.2f}$")

        ax.set_title(f"$n = {n}$ Bidders (Equilibrium $k^*={eq_k:.3f}$)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Deviation Factor $k$ ($b_1 = k \\cdot v_1$)", fontsize=9.5)
        ax.set_ylabel("Expected Utility $\\mathbb{E}[u_1]$", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "part2_utility.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out_path}")


def plot_overpaid(results: dict):
    """Plot overpayment metrics: fraction and regret amount."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle("Part 2: Analysis of 'Winner Could Have Paid Less and Still Won'\n"
                 "Structural Overpayment (Paying > 2nd bid) and Regret Margin in First-Price Auctions",
                 fontsize=13, fontweight="bold", y=0.99)

    for col_idx, n in enumerate(N_VALUES):
        k_vals = results[n]["k_grid"]
        eq_k = results[n]["eq_k"]

        ax_top = axes[0, col_idx]
        ax_top.plot(k_vals, results[n]["op_all"], "s-", color="#9467bd", markersize=3.5, linewidth=1.8,
                    label="All Auctions Winner")
        ax_top.plot(k_vals, results[n]["op_b1"], "^--", color="#d62728", markersize=3.5, linewidth=1.5,
                    label="Bidder 1 (given win)")
        ax_top.axvline(eq_k, color="black", linestyle="-.", linewidth=1.4, label=f"BNE $k={eq_k:.3f}$")
        ax_top.set_title(f"$n = {n}$: Fraction Winner Paid > 2nd Bid", fontsize=10.5, fontweight="bold")
        ax_top.set_xlabel("Scaling Factor $k$", fontsize=9)
        ax_top.set_ylabel("Fraction (1.0 = 100%)", fontsize=9)
        ax_top.set_ylim(0.95, 1.02)
        ax_top.legend(fontsize=7.5, loc="lower right")
        ax_top.grid(True, linestyle="--", alpha=0.4)

        ax_bot = axes[1, col_idx]
        ax_bot.plot(k_vals, results[n]["regret_all"], "d-", color="#8c564b", markersize=3.5, linewidth=1.8,
                    label="Overall $\\mathbb{E}[b_{(1)} - b_{(2)}]$")
        ax_bot.plot(k_vals, results[n]["regret_b1"], "o--", color="#e377c2", markersize=3.5, linewidth=1.5,
                    label="Bidder 1 $\\mathbb{E}[b_1 - b_{rival}^{max} \\mid \\mathrm{win}]$")
        ax_bot.axvline(eq_k, color="black", linestyle="-.", linewidth=1.4, label=f"BNE $k={eq_k:.3f}$")
        ax_bot.set_title(f"$n = {n}$: Overpayment Margin $\\mathbb{E}[b_{(1)} - b_{(2)}]$", fontsize=10.5, fontweight="bold")
        ax_bot.set_xlabel("Scaling Factor $k$", fontsize=9)
        ax_bot.set_ylabel("Money Left on Table ($b_{win} - b_{second}$)", fontsize=9)
        ax_bot.legend(fontsize=7.5, loc="upper left")
        ax_bot.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PLOTS_DIR, "part2_overpaid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out_path}")


def print_summary(results: dict):
    """Print academic-grade summary table of Part 2 results."""
    print("\n" + "=" * 80)
    print("  PART 2: FIRST-PRICE AUCTION EQUILIBRIUM & OVERPAYMENT SUMMARY")
    print("=" * 80)
    print(f"{'n':<4} {'BNE k*':<8} {'Sim Peak k':<11} {'BNE Utility':<13} {'Peak Utility':<13} {'Win % @ BNE':<12} {'Overpayment'}")
    print("-" * 80)
    for n in N_VALUES:
        k_vals = results[n]["k_grid"]
        eq_k = results[n]["eq_k"]
        eq_idx = np.argmin(np.abs(k_vals - eq_k))
        best_idx = np.argmax(results[n]["utilities"])

        bne_u = results[n]["utilities"][eq_idx]
        peak_u = results[n]["utilities"][best_idx]
        bne_k = eq_k
        peak_k = k_vals[best_idx]
        win_p = results[n]["win_probs"][eq_idx] * 100.0
        regret = results[n]["regret_all"][eq_idx]

        print(f"{n:<4} {bne_k:<8.3f} {peak_k:<11.3f} {bne_u:<13.5f} {peak_u:<13.5f} {win_p:<11.1f}% {regret:.4f}")
    print("=" * 80)
    print("CRITICAL CLARIFICATION ON OVERPAYMENT:")
    print("  • The 100.0% overpayment result means the winner pays strictly more than the")
    print("    second-highest bid (b_(1) > b_(2)) due to continuous distributions without ties.")
    print("  • This represents money left on the table (ex-post regret), NOT that the winner")
    print("    makes a loss. At equilibrium k* = (n-1)/n < 1, the winner earns positive surplus.")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Part 2: First-Price Auction Simulation & Overpayment Analysis")
    print("=" * 70)
    results = run_experiment()

    print("\n[Generating Plots]")
    plot_utility(results)
    plot_overpaid(results)

    print_summary(results)
    print("Part 2 complete. Plots saved:")
    print("  - plots/part2_utility.png")
    print("  - plots/part2_overpaid.png")


if __name__ == "__main__":
    main()
