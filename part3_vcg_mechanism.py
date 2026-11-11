"""
Part 3: VCG Mechanism for Cheapest Path Auction
================================================
Implements the Vickrey-Clarke-Groves (VCG) mechanism for a network path
auction problem. The goal is to find the cheapest path from A to F and
compute truthful payments for each edge on the optimal path.

Graph (Figure 7.3 style from Nisan et al. / auction theory textbook):
  Nodes: A, B, C, D, E, F
  TRUE edge costs: AB=1, AC=4, BD=1, BE=2, CD=1, DE=1, DF=4, EF=1
  Optimal path: A→B→E→F (cost 4)

VCG Payments (reproduced exactly):
  AB = 4, BE = 2, EF = 3  (all other edges = 0)
"""

import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# TRUE costs (fixed throughout — not revealed to the mechanism)
TRUE_COSTS = {
    ('A', 'B'): 1,
    ('A', 'C'): 4,
    ('B', 'D'): 1,
    ('B', 'E'): 2,
    ('C', 'D'): 1,
    ('D', 'E'): 1,
    ('D', 'F'): 4,
    ('E', 'F'): 1,
}

# Canonical edge ordering (used for display / iteration)
EDGES = list(TRUE_COSTS.keys())


def build_graph(cost_dict):
    """Build an undirected NetworkX graph from a {(u,v): weight} dict."""
    G = nx.Graph()
    for (u, v), w in cost_dict.items():
        G.add_edge(u, v, weight=w)
    return G


def get_edge_cost(cost_dict, u, v):
    """Return edge cost regardless of direction."""
    return cost_dict.get((u, v), cost_dict.get((v, u), None))


def shortest_path_cost(G, source='A', target='F'):
    """Return (path_node_list, total_cost) or (None, inf) if disconnected."""
    try:
        path = nx.shortest_path(G, source, target, weight='weight')
        cost = nx.shortest_path_length(G, source, target, weight='weight')
        return path, cost
    except nx.NetworkXNoPath:
        return None, float('inf')


def path_edges(path):
    """Convert a node path to a list of canonical (u,v) edge tuples."""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def canonical(u, v):
    """Return edge in canonical (alphabetical) order."""
    return (u, v) if u < v else (v, u)


# ---------------------------------------------------------------------------
# VCG computation
# ---------------------------------------------------------------------------

def compute_vcg(cost_dict, source='A', target='F'):
    """
    Compute VCG payments for all edges.

    Returns
    -------
    optimal_path : list of nodes
    d_opt        : float, total cost of optimal path
    payments     : dict  {canonical_edge: payment}
    utilities    : dict  {canonical_edge: utility = payment - true_cost}
    """
    G = build_graph(cost_dict)
    opt_path, d_opt = shortest_path_cost(G, source, target)
    if opt_path is None:
        raise ValueError(f"No path from {source} to {target}")

    opt_edges = path_edges(opt_path)
    payments = {}
    utilities = {}

    for (u, v), true_cost in cost_dict.items():
        edge_canon = canonical(u, v)

        # Check if this edge is on the optimal path
        on_path = any(canonical(a, b) == edge_canon for a, b in opt_edges)

        if not on_path:
            payments[edge_canon] = 0.0
            utilities[edge_canon] = 0.0 - true_cost  # usually 0 cost assumed for off-path
        else:
            # VCG payment: d(A,F without e) - [d_opt - cost(e)]
            G_minus_e = G.copy()
            G_minus_e.remove_edge(u, v)
            _, d_no_e = shortest_path_cost(G_minus_e, source, target)

            cost_e = get_edge_cost(cost_dict, u, v)
            payment_e = d_no_e - (d_opt - cost_e)
            payments[edge_canon] = payment_e
            utilities[edge_canon] = payment_e - true_cost

    return opt_path, d_opt, payments, utilities


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(opt_path, d_opt, payments, utilities):
    """Print a formatted summary of the VCG auction results."""
    opt_edges_set = {canonical(a, b) for a, b in path_edges(opt_path)}

    header = f"{'Edge':<8} {'True Cost':>10} {'On Path':>8} {'VCG Pay':>10} {'Utility':>10}"
    sep = '-' * len(header)
    print()
    print("=" * len(header))
    print("  VCG MECHANISM — CHEAPEST PATH AUCTION  (A → F)")
    print("=" * len(header))
    print(f"  Optimal path : {' → '.join(opt_path)}   (total cost = {d_opt})")
    print("=" * len(header))
    print(header)
    print(sep)

    for (u, v) in EDGES:
        edge_canon = canonical(u, v)
        true_cost = TRUE_COSTS[(u, v)]
        on_path = "Yes" if edge_canon in opt_edges_set else "No"
        pay = payments.get(edge_canon, 0.0)
        util = utilities.get(edge_canon, -true_cost)
        print(f"  ({u},{v})  {true_cost:>10}   {on_path:>7}  {pay:>10.2f}  {util:>10.2f}")

    print(sep)
    total_payment = sum(p for e, p in payments.items() if e in opt_edges_set)
    print(f"  {'TOTAL PAYMENT':>30}  {total_payment:>10.2f}")
    print("=" * len(header))
    print()

    # Verification
    print("Verification against textbook targets:")
    targets = {('A', 'B'): 4.0, ('B', 'E'): 2.0, ('E', 'F'): 3.0}
    all_ok = True
    for (u, v), target_pay in targets.items():
        edge_canon = canonical(u, v)
        actual = payments.get(edge_canon, 0.0)
        status = "✓" if abs(actual - target_pay) < 1e-9 else "✗"
        print(f"  {status}  payment({u},{v}) = {actual:.1f}  (target {target_pay:.1f})")
        if abs(actual - target_pay) >= 1e-9:
            all_ok = False
    print()
    if all_ok:
        print("  ✓ All payments match textbook Figure 7.3!\n")
    else:
        print("  ✗ Some payments do not match targets.\n")


# ---------------------------------------------------------------------------
# Misreporting analysis helper
# ---------------------------------------------------------------------------

def run_vcg_with_reported_cost(edge_uv, reported_cost, source='A', target='F'):
    """
    Re-run VCG replacing the reported cost of `edge_uv` with `reported_cost`.
    Returns (payment_for_edge, is_on_path, optimal_path).
    """
    cost_dict = dict(TRUE_COSTS)
    # Replace reported cost (keep true cost separate)
    cost_dict[edge_uv] = reported_cost

    G = build_graph(cost_dict)
    opt_path, d_opt = shortest_path_cost(G, source, target)
    if opt_path is None:
        return 0.0, False, []

    opt_edges_set = {canonical(a, b) for a, b in path_edges(opt_path)}
    edge_canon = canonical(*edge_uv)
    on_path = edge_canon in opt_edges_set

    if not on_path:
        payment = 0.0
    else:
        G_minus_e = G.copy()
        G_minus_e.remove_edge(*edge_uv)
        _, d_no_e = shortest_path_cost(G_minus_e, source, target)
        cost_e = reported_cost
        payment = d_no_e - (d_opt - cost_e)

    return payment, on_path, opt_path


# ---------------------------------------------------------------------------
# Misreporting plots
# ---------------------------------------------------------------------------

def plot_misreporting(save_path='plots/part3_vcg_misreporting.png'):
    """
    Plot 2×2 grid analysing misreporting incentives for:
      - Row 1: edge ON  the optimal path  (AB, true cost 1)
      - Row 2: edge OFF the optimal path  (AC, true cost 4)
    Columns:
      - Col 1: utility vs reported cost
      - Col 2: path inclusion vs reported cost
    """
    # First run baseline to know the optimal path
    opt_path, d_opt, payments, utilities = compute_vcg(TRUE_COSTS)
    opt_edges_set = {canonical(a, b) for a, b in path_edges(opt_path)}

    # Select example edges
    edge_on  = ('A', 'B')   # on optimal path,  true cost = 1
    edge_off = ('A', 'C')   # off optimal path, true cost = 4

    reported_costs = np.arange(0, 10.5, 0.5)   # 0, 0.5, 1.0, …, 10.0  (21 values)

    # Collect data
    results = {}
    for edge in [edge_on, edge_off]:
        true_cost = TRUE_COSTS[edge]
        utils, on_path_flags = [], []
        for rc in reported_costs:
            payment, on_path, _ = run_vcg_with_reported_cost(edge, rc)
            utility = payment - true_cost   # utility = payment - TRUE cost (fixed)
            utils.append(utility)
            on_path_flags.append(1 if on_path else 0)
        results[edge] = {
            'true_cost': true_cost,
            'utilities': utils,
            'on_path': on_path_flags,
        }

    # ------------------------------------------------------------------ plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('VCG Mechanism — Misreporting Incentive Analysis\n'
                 '(Optimal path: A→B→E→F, payments: AB=4, BE=2, EF=3)',
                 fontsize=13, fontweight='bold')

    row_labels  = ['Edge ON path (AB, true cost = 1)',
                   'Edge OFF path (AC, true cost = 4)']
    col_labels  = ['Utility vs Reported Cost', 'Path Inclusion vs Reported Cost']
    edge_colors = ['steelblue', 'darkorange']

    for row, edge in enumerate([edge_on, edge_off]):
        true_cost   = results[edge]['true_cost']
        utils       = results[edge]['utilities']
        on_path_flg = results[edge]['on_path']
        color       = edge_colors[row]

        # --- Col 0: Utility
        ax = axes[row][0]
        ax.plot(reported_costs, utils, color=color, lw=2.2, marker='o',
                markersize=4, label='Utility')
        ax.axvline(true_cost, color='red', ls='--', lw=1.8,
                   label=f'Truthful report (c={true_cost})')
        ax.axhline(0, color='gray', ls=':', lw=1.0)

        # Mark truthful utility
        # Find closest reported cost to true cost
        idx_truth = np.argmin(np.abs(reported_costs - true_cost))
        ax.scatter([reported_costs[idx_truth]], [utils[idx_truth]],
                   color='red', zorder=5, s=80, label=f'Truthful utility')

        ax.set_title(f'{row_labels[row]}\n{col_labels[0]}', fontsize=10)
        ax.set_xlabel('Reported cost')
        ax.set_ylabel('Utility (payment − true cost)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Col 1: Path inclusion
        ax = axes[row][1]
        ax.step(reported_costs, on_path_flg, color=color, lw=2.2, where='post',
                label='On optimal path')
        ax.axvline(true_cost, color='red', ls='--', lw=1.8,
                   label=f'Truthful report (c={true_cost})')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.set_ylim(-0.15, 1.3)

        # Find transition points
        transitions = np.where(np.diff(on_path_flg) != 0)[0]
        for t in transitions:
            xval = reported_costs[t + 1]
            ax.axvline(xval, color='purple', ls=':', lw=1.5,
                       label=f'Enters/exits path at c≈{xval:.1f}')

        ax.set_title(f'{row_labels[row]}\n{col_labels[1]}', fontsize=10)
        ax.set_xlabel('Reported cost')
        ax.set_ylabel('Included in optimal path?')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Misreporting plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  PART 3: VCG MECHANISM — CHEAPEST PATH AUCTION")
    print("=" * 60)

    # ------- Step 1: Run VCG with true costs ---------------------------------
    print("\n[1] Computing VCG with true costs …")
    opt_path, d_opt, payments, utilities = compute_vcg(TRUE_COSTS)

    print(f"    Graph edges (true costs): {dict(TRUE_COSTS)}")
    print(f"    Optimal path: {' → '.join(opt_path)}  (cost = {d_opt})")

    # ------- Step 2: Print summary table -------------------------------------
    print_summary_table(opt_path, d_opt, payments, utilities)

    # ------- Step 3: Misreporting analysis -----------------------------------
    print("[2] Running misreporting analysis …")
    os.makedirs('plots', exist_ok=True)
    plot_misreporting('plots/part3_vcg_misreporting.png')

    # ------- Step 4: Brief narrative -----------------------------------------
    print("\n[3] Key takeaways:")
    print("    • VCG makes truthful reporting a dominant strategy.")
    print("    • Edge AB (on path, cost 1): misreporting a higher cost")
    print("      eventually causes exclusion from the path → 0 payment but")
    print("      utility = 0 - 1 = -1 (worse than truthful utility = 3).")
    print("    • Edge AC (off path, cost 4): regardless of reported cost,")
    print("      it cannot improve its utility beyond 0 by misreporting;")
    print("      underreporting to get on path yields negative utility.")
    print("    • Both edges are best-off reporting truthfully ⇒ incentive-")
    print("      compatible mechanism.\n")


if __name__ == "__main__":
    main()
