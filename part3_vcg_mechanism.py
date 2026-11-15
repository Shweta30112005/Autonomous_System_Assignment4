"""
Part 3: VCG Mechanism for Shortest-Path Procurement Auction
===========================================================
Autonomous Systems - Assignment 4
Authors: Akanksha Singh (B23CM1005), Shweta Mandal (B23CM1037)

Implements the Vickrey-Clarke-Groves (VCG) mechanism with the Clarke pivot rule
for purchasing the cheapest path from source node A to destination node F
in a communication network where each edge is an independent self-interested agent
possessing a private cost.

Based on Figure 7.3 from:
  Yoav Shoham and Kevin Leyton-Brown,
  "Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations"
  Cambridge University Press.

True Edge Costs:
  AB = 1, AC = 4, BD = 1, BE = 2, CD = 1, DE = 1, DF = 4, EF = 1

Optimal Path:
  A -> B -> E -> F (Total cost = 1 + 2 + 1 = 4)

Theoretical VCG Payments:
  p(AB) = 4, p(BE) = 2, p(EF) = 3; all other edges receive 0.

Individual Rationality and Incentive Compatibility:
  u_e = p_e - c_e^true if edge e is chosen (on optimal path)
  u_e = 0.0            if edge e is NOT chosen (off optimal path)
"""

import os
import tempfile

# Prevent matplotlib cache warning
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"

# TRUE private costs of all edges (Figure 7.3)
TRUE_COSTS = {
    ("A", "B"): 1.0,
    ("A", "C"): 4.0,
    ("B", "D"): 1.0,
    ("B", "E"): 2.0,
    ("C", "D"): 1.0,
    ("D", "E"): 1.0,
    ("D", "F"): 4.0,
    ("E", "F"): 1.0,
}

EDGES = list(TRUE_COSTS.keys())


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def canonical(u: str, v: str) -> tuple[str, str]:
    """Return edge tuple in alphabetical canonical order."""
    return (u, v) if u < v else (v, u)


def build_graph(cost_dict: dict) -> nx.Graph:
    """Build an undirected NetworkX graph from edge cost mapping."""
    G = nx.Graph()
    for (u, v), w in cost_dict.items():
        G.add_edge(u, v, weight=w)
    return G


def shortest_path_cost(G: nx.Graph, source="A", target="F") -> tuple[list, float]:
    """
    Find shortest path from source to target with deterministic tie-breaking:
      1. Primary key: Minimal total path cost
      2. Secondary key: Minimal hop-count (fewest edges)
      3. Tertiary key: Lexicographical node sequence

    This guarantees:
      - Baseline: ABEF (cost 4.0, 3 hops) is chosen over ABDEF (cost 4.0, 4 hops),
        faithfully reproducing textbook payments p(AB)=4, p(BE)=2, p(EF)=3.
      - At misreport c'(AC) = 1.0: ABEF (cost 4.0, 3 hops) beats ACDEF (cost 4.0, 4 hops).
    """
    try:
        min_cost = nx.shortest_path_length(G, source, target, weight="weight")
    except nx.NetworkXNoPath:
        return None, float("inf")

    candidates = list(nx.all_shortest_paths(G, source, target, weight="weight"))
    # Sort candidates deterministically: (hop_count, node_sequence)
    candidates.sort(key=lambda p: (len(p), p))
    return candidates[0], float(min_cost)


def path_to_edges(path: list) -> list[tuple[str, str]]:
    """Convert node sequence [n1, n2, n3] to canonical edge list."""
    if not path or len(path) < 2:
        return []
    return [canonical(path[i], path[i + 1]) for i in range(len(path) - 1)]


# ---------------------------------------------------------------------------
# VCG Mechanism Implementation
# ---------------------------------------------------------------------------

def compute_vcg(cost_dict: dict, source="A", target="F"):
    """
    Compute VCG payments and utilities using the Clarke pivot rule.

    Formula:
      For winning edge e on optimal path:
        p_e = d(G - {e}) - [d(G) - c_e^reported]
        u_e = p_e - c_e^true
      For unselected edge e (off optimal path):
        p_e = 0.0
        u_e = 0.0  (No cost incurred since edge is not utilized)

    Returns:
      opt_path: list of node names along optimal path
      d_opt: float total cost of optimal path
      payments: dict {canonical_edge: payment}
      utilities: dict {canonical_edge: utility}
    """
    G = build_graph(cost_dict)
    opt_path, d_opt = shortest_path_cost(G, source, target)
    if opt_path is None:
        raise ValueError(f"No path between {source} and {target}")

    opt_edges_set = set(path_to_edges(opt_path))
    payments = {}
    utilities = {}

    for (u, v), rep_cost in cost_dict.items():
        edge_canon = canonical(u, v)
        true_c = TRUE_COSTS[edge_canon]

        if edge_canon not in opt_edges_set:
            payments[edge_canon] = 0.0
            utilities[edge_canon] = 0.0
        else:
            G_without_e = G.copy()
            G_without_e.remove_edge(u, v)
            _, d_no_e = shortest_path_cost(G_without_e, source, target)

            p_e = d_no_e - (d_opt - rep_cost)
            payments[edge_canon] = float(p_e)
            utilities[edge_canon] = float(p_e - true_c)

    return opt_path, d_opt, payments, utilities


def run_vcg_with_reported_cost(edge_uv: tuple[str, str], reported_cost: float,
                               source="A", target="F"):
    """
    Re-evaluate VCG auction when edge_uv deviates by reporting reported_cost,
    while all other edges report truthfully.
    """
    edge_canon = canonical(*edge_uv)
    true_cost = TRUE_COSTS[edge_canon]

    rep_dict = dict(TRUE_COSTS)
    rep_dict[edge_canon] = reported_cost

    G = build_graph(rep_dict)
    opt_path, d_opt = shortest_path_cost(G, source, target)
    if opt_path is None:
        return 0.0, 0.0, False, []

    opt_edges_set = set(path_to_edges(opt_path))
    on_path = edge_canon in opt_edges_set

    if not on_path:
        payment = 0.0
        utility = 0.0
    else:
        G_without_e = G.copy()
        G_without_e.remove_edge(*edge_uv)
        _, d_no_e = shortest_path_cost(G_without_e, source, target)
        payment = d_no_e - (d_opt - reported_cost)
        utility = payment - true_cost

    return float(payment), float(utility), on_path, opt_path


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_misreporting():
    """
    Examine misreported costs from 0.0 to 10.0 in steps of 0.5:
      - Edge ON path:  AB (true cost = 1.0)
      - Edge OFF path: AC (true cost = 4.0)
    Generates 2x2 grid showing Utility and Path Inclusion vs Reported Cost.
    """
    ensure_plots_dir()
    edge_on = ("A", "B")
    edge_off = ("A", "C")
    reported_grid = np.arange(0.0, 10.5, 0.5)

    data = {}
    for edge in [edge_on, edge_off]:
        edge_canon = canonical(*edge)
        true_c = TRUE_COSTS[edge_canon]
        utils, pays, on_path_flags = [], [], []

        for rep_c in reported_grid:
            p, u, on_p, _ = run_vcg_with_reported_cost(edge, rep_c)
            utils.append(u)
            pays.append(p)
            on_path_flags.append(1 if on_p else 0)

        data[edge_canon] = {
            "true_cost": true_c,
            "utilities": np.array(utils),
            "payments": np.array(pays),
            "on_path": np.array(on_path_flags),
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.suptitle(
        "Part 3: VCG Mechanism — Misreporting Incentive Compatibility Analysis\n"
        "Testing Strategic Deviations from 0.0 to 10.0 (Step = 0.5)",
        fontsize=13, fontweight="bold", y=0.99
    )

    configs = [
        (edge_on, "Edge ON Path: AB (True Cost $c_{AB} = 1.0$)", "#1f77b4", 0),
        (edge_off, "Edge OFF Path: AC (True Cost $c_{AC} = 4.0$)", "#d95f02", 1),
    ]

    for edge, title_str, color, row in configs:
        edge_canon = canonical(*edge)
        true_c = data[edge_canon]["true_cost"]
        utils = data[edge_canon]["utilities"]
        on_p = data[edge_canon]["on_path"]

        # Col 0: Utility vs Reported Cost
        ax_u = axes[row][0]
        ax_u.plot(reported_grid, utils, "o-", color=color, linewidth=2.0, markersize=5,
                  label="Agent Utility $u_e(c')$")
        ax_u.axvline(true_c, color="forestgreen", linestyle="--", linewidth=2.0,
                     label=f"Truthful Cost $c_e={true_c:.1f}$")
        ax_u.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)

        truth_idx = np.argmin(np.abs(reported_grid - true_c))
        ax_u.scatter([reported_grid[truth_idx]], [utils[truth_idx]], color="forestgreen",
                     s=90, zorder=6, label=f"Truthful $u={utils[truth_idx]:.1f}$")

        ax_u.set_title(f"{title_str}\nUtility vs Reported Cost", fontsize=10.5, fontweight="bold")
        ax_u.set_xlabel("Reported Cost $c_e'$", fontsize=9.5)
        ax_u.set_ylabel("Utility $u_e = p_e - c_e^{\\mathrm{true}}$", fontsize=9.5)
        ax_u.legend(fontsize=8, loc="best")
        ax_u.grid(True, linestyle="--", alpha=0.4)

        # Col 1: Path Inclusion vs Reported Cost
        ax_p = axes[row][1]
        ax_p.step(reported_grid, on_p, where="post", color=color, linewidth=2.2,
                  label="Path Inclusion (1=Yes, 0=No)")
        ax_p.axvline(true_c, color="forestgreen", linestyle="--", linewidth=2.0,
                     label=f"Truthful Cost $c_e={true_c:.1f}$")
        ax_p.set_yticks([0, 1])
        ax_p.set_yticklabels(["Excluded (No)", "Selected (Yes)"])
        ax_p.set_ylim(-0.15, 1.25)

        diffs = np.diff(on_p)
        transitions = np.where(diffs != 0)[0]
        for t in transitions:
            switch_pt = reported_grid[t + 1]
            ax_p.axvline(switch_pt, color="purple", linestyle=":", linewidth=1.8,
                         label=f"Transition at $c' \\approx {switch_pt:.1f}$")

        ax_p.set_title(f"{title_str}\nPath Selection vs Reported Cost", fontsize=10.5, fontweight="bold")
        ax_p.set_xlabel("Reported Cost $c_e'$", fontsize=9.5)
        ax_p.set_ylabel("Selection Status", fontsize=9.5)
        ax_p.legend(fontsize=8, loc="best")
        ax_p.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(PLOTS_DIR, "part3_misreporting.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out_path}")


def print_vcg_table(opt_path: list, d_opt: float, payments: dict, utilities: dict):
    """Print academically structured VCG outcome table."""
    opt_edges_set = set(path_to_edges(opt_path))

    header = f"{'Edge':<10} {'True Cost':>10} {'Reported':>10} {'Selected?':>11} {'VCG Payment':>13} {'Utility':>10}"
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  VCG MECHANISM: SHORTEST PATH PROCUREMENT AUCTION (FIGURE 7.3)")
    print("=" * len(header))
    print(f"  Source Node: A | Destination Node: F")
    print(f"  Chosen Optimal Path: {' -> '.join(opt_path)} (Total Cost = {d_opt:.1f})")
    print("=" * len(header))
    print(header)
    print(sep)

    for edge in EDGES:
        edge_canon = canonical(*edge)
        tc = TRUE_COSTS[edge_canon]
        sel = "YES" if edge_canon in opt_edges_set else "NO"
        p = payments[edge_canon]
        u = utilities[edge_canon]
        print(f"  {str(edge):<8} {tc:>10.1f} {tc:>10.1f} {sel:>11} {p:>13.2f} {u:>10.2f}")

    print(sep)
    total_pay = sum(payments.values())
    print(f"  {'TOTAL MECHANISM PAYMENT:':>45} {total_pay:>13.2f}")
    print("=" * len(header))

    targets = {("A", "B"): 4.0, ("B", "E"): 2.0, ("E", "F"): 3.0}
    print("\nTextbook Benchmark Verification:")
    all_matched = True
    for edge, target_p in targets.items():
        actual_p = payments[edge]
        match = np.isclose(actual_p, target_p, atol=1e-5)
        status = "PASSED [OK]" if match else "FAILED [MISMATCH]"
        print(f"  * Edge {edge}: Target = {target_p:.1f}, Computed = {actual_p:.1f} -> {status}")
        if not match:
            all_matched = False

    if all_matched:
        print("\n  ALL VCG PAYMENTS PERFECTLY REPRODUCE TEXTBOOK FIGURE 7.3 TARGETS!\n")


def print_misreporting_summary():
    """Print analytical breakdown of strategic misreporting incentives and tie-breaking."""
    print("=" * 70)
    print("  STRATEGIC MISREPORTING INCENTIVE & TIE-BREAKING ANALYSIS")
    print("=" * 70)
    print("1. Baseline Graph Tie-Breaking (ABEF vs ABDEF):")
    print("   - Both paths ABEF (1+2+1=4.0) and ABDEF (1+1+1+1=4.0) have total cost 4.0.")
    print("   - Implementation uses deterministic minimal hop-count tie-breaking:")
    print("     ABEF has 3 edges (4 nodes) vs ABDEF with 4 edges (5 nodes).")
    print("   - Minimal hop-count selects ABEF, perfectly reproducing textbook payments:")
    print("     p(AB) = 4, p(BE) = 2, p(EF) = 3, and all other edges = 0.")
    print("\n2. Winning Edge AB (True Cost = 1.0, Truthful Payment = 4.0, Utility = 3.0):")
    print("   - Reporting c' in [0.0, 4.0]: AB remains on the path.")
    print("     Because payment depends solely on others' costs, payment remains 4.0")
    print("     and utility remains constant at 3.0.")
    print("   - Reporting c' > 4.0: Alternative path ACDEF (cost 7) becomes cheaper.")
    print("     AB is excluded from the path, receiving payment 0.0 and utility 0.0.")
    print("   -> Conclusion: No misreport can increase AB's utility beyond 3.0.")
    print("\n3. Losing Edge AC (True Cost = 4.0, Truthful Payment = 0.0, Utility = 0.0):")
    print("   - Reporting c' > 1.0: Path ACDEF cost > 4.0; AC remains excluded (u = 0.0).")
    print("   - Reporting c' = 1.0 (Boundary Tie Condition):")
    print("     Path ACDEF cost = 1.0 + 3 = 4.0, tying with ABEF (cost 4.0).")
    print("     * Under minimal hop-count tie-breaking (3 hops vs 4 hops), ABEF wins,")
    print("       so AC is excluded and gets utility 0.0.")
    print("     * Under an alternative tie-breaker selecting ACDEF, AC's VCG payment is:")
    print("       p = d(G - {AC}) - [d(G) - c'] = 4.0 - 3.0 = 1.0.")
    print("       AC's utility would be p - c_true = 1.0 - 4.0 = -3.0 (severe loss!).")
    print("   - Underreporting c' < 1.0: AC enters path ACDEF (cost c' + 3 < 4).")
    print("     VCG payment is p = 4.0 - 3.0 = 1.0; utility is 1.0 - 4.0 = -3.0 < 0.")
    print("   -> Conclusion: AC achieves maximum utility (0.0) by reporting truthfully.")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Part 3: VCG Mechanism for Shortest-Path Procurement")
    print("=" * 70)

    opt_path, d_opt, payments, utilities = compute_vcg(TRUE_COSTS)
    print_vcg_table(opt_path, d_opt, payments, utilities)

    print("\n[Running Strategic Misreporting Analysis (Grid: 0.0 to 10.0, step 0.5)]")
    plot_misreporting()

    print_misreporting_summary()
    print("Part 3 complete. Plot saved:")
    print("  - plots/part3_misreporting.png")


if __name__ == "__main__":
    main()
