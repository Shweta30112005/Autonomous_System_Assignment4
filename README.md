# Assignment 4: Mechanism Design & Auction Theory

## Overview

This assignment implements three classic mechanism design problems from auction theory:

1. **Part 1** — Second-Price (Vickrey) Sealed-Bid Auction
2. **Part 2** — First-Price Auction with Equilibrium Deviations
3. **Part 3** — VCG Mechanism for Shortest-Path Procurement

---

## Project Structure

```
.
├── README.md
├── part1_second_price_auction.py   # Second-price auction simulation
├── part2_first_price_auction.py    # First-price auction with deviations
├── part3_vcg_mechanism.py          # VCG mechanism for cheapest path
└── plots/                          # All generated figures
    ├── part1A_one_rival.png        # Part 1A: 1 rival, 9 subplots
    ├── part1B_fixed_rival.png      # Part 1B: Fixed rival bids
    ├── part1C_two_rivals.png       # Part 1C: 2 rivals, 9 subplots
    ├── part2_first_price_utility.png   # Part 2: Utility vs k
    ├── part2_first_price_overpaid.png  # Part 2: Overpaid fraction vs k
    ├── part2_combined.png          # Part 2: Combined figure
    └── part3_vcg_misreporting.png  # Part 3: VCG misreporting analysis
```

---

## Dependencies

```bash
pip install numpy matplotlib networkx
```

---

## Running the Code

```bash
# Run each part independently
python part1_second_price_auction.py
python part2_first_price_auction.py
python part3_vcg_mechanism.py
```

---

## Part 1: Second-Price Sealed-Bid Auction

### Setup
- **Bidder 1**: Fixed value `v ∈ {0.2, 0.5, 0.8}`; bids truthfully at the optimal
- **Rival bidder(s)**: Value drawn from `Uniform[0,1]`
- **Rival strategies**:
  - (a) Truthful: `b₂ = v₂`
  - (b) Half-value: `b₂ = v₂/2`
  - (c) Random: `b₂ ~ Uniform[0,1]` (ignores true value)

### Auction Rule
- Bidder 1 wins if `b₁ > b₂` (or `> max(b₂, b₃)` with 2 rivals)
- If win: utility = `v₁ - b₂` (pays second-highest bid)
- If lose: utility = 0

### Part A — 1 Rival
Sweeps `b₁ ∈ [0.00, 1.00]` in steps of 0.01, runs 100,000 simulations per point.
9 subplots (3 values × 3 strategies).

**Key finding**: The optimal bid is always `b₁ = v₁` (truthful bidding), regardless of the rival's strategy.

### Part B — Fixed Rival Bid
With rival bid fixed at `{0.1, 0.4, 0.9}`, analytically confirms `b₁ = v₁` maximises utility.
Utility is a step function: 0 when `b₁ ≤ fixed_rival_bid`, else `v₁ - fixed_rival_bid`.

### Part C — 2 Rivals
Same as Part A with two rivals; bidder 1 wins only if `b₁ > max(b₂, b₃)`.

---

## Part 2: First-Price Auction

### Setup
- **n** bidders; `n ∈ {2, 3, 4}`
- Bidder 1 bids `k · v₁` where `k ∈ {0.1, 0.2, ..., 2.0}`
- Rivals `2..n` bid equilibrium: `((n-1)/n) · vᵢ`, `vᵢ ~ Uniform[0,1]`
- Winner = highest bid; pays own bid (first-price)
- Bidder 1 utility = `v₁ - k·v₁ = v₁(1-k)` if win, else 0

### Key Results
- Equilibrium `k = (n-1)/n` maximises expected utility
- Fraction of "overpaid" auctions: auctions where bidder 1 wins but could have bid lower

---

## Part 3: VCG Mechanism — Cheapest Path

### Graph (Figure 7.3)
Nodes: A, B, C, D, E, F  
Edges and true costs reproduced from the textbook.

### VCG Payments
Optimal path: **A → B → E → F**  
Payments: AB=4, BE=2, EF=3; all other edges=0

### VCG Payment Formula
For each edge `e` on the optimal path:
```
payment(e) = cost_of_opt_path_without_e - (total_opt_path_cost - true_cost(e))
```

### Misreporting Analysis
- Edge **on** the optimal path (e.g. AB): increasing reported cost can price the edge out
- Edge **off** the optimal path (e.g. CD): decreasing reported cost can bring it onto the path

**Key finding**: Truthful reporting maximises each edge agent's utility (VCG is incentive-compatible).
