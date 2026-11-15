# Autonomous Systems - Assignment 4: Mechanism Design & Auction Theory

---

## Table of Contents
1. [Executive Summary & Objectives](#1-executive-summary--objectives)
2. [Theoretical Foundations](#2-theoretical-foundations)
   - [2.1 Second-Price (Vickrey) Sealed-Bid Auction](#21-second-price-vickrey-sealed-bid-auction)
   - [2.2 First-Price Sealed-Bid Auction & Bayesian Nash Equilibrium](#22-first-price-sealed-bid-auction--bayesian-nash-equilibrium)
   - [2.3 The Vickrey-Clarke-Groves (VCG) Mechanism](#23-the-vickrey-clarke-groves-vcg-mechanism)
3. [Repository Structure](#3-repository-structure)
4. [Installation & Requirements](#4-installation--requirements)
5. [Execution Instructions](#5-execution-instructions)
6. [Detailed Experimental Results & Analysis](#6-detailed-experimental-results--analysis)
   - [Part 1: Second-Price Sealed-Bid Auction](#part-1-second-price-sealed-bid-auction)
   - [Part 2: First-Price Sealed-Bid Auction](#part-2-first-price-sealed-bid-auction)
   - [Part 3: VCG Mechanism for Shortest-Path Procurement](#part-3-vcg-mechanism-for-shortest-path-procurement)
7. [Computational Complexity & Scalability Analysis](#7-computational-complexity--scalability-analysis)
8. [Limitations & Practical Considerations](#8-limitations--practical-considerations)
9. [Conclusions](#9-conclusions)
10. [References](#10-references)

---

## 1. Executive Summary & Objectives

In multi-agent systems, autonomous agents frequently possess private information such as task completion costs, valuation of shared resources, or latency bounds and seek to optimize their individual payoff. If the system protocol relies on naive reporting, rational self-interested agents have strong incentives to misreport their private data to manipulate the outcome in their favor.

**Mechanism design** resolves this challenge by formulating game-theoretic rules and transfer payment functions that align individual selfish incentives with socially optimal system objectives.

This assignment explores three fundamental mechanism design frameworks:
1. **Part 1 (Second-Price Sealed-Bid Auction):** Simulates $N = 100{,}000$ auctions per candidate bid across valuations $v_1 \in \{0.2, 0.5, 0.8\}$ against rivals adopting truthful, half-value, and random bidding strategies. Demonstrates that truthful bidding ($b_1 = v_1$) is weakly dominant, identifies the half-value plateau phenomenon, and analytically verifies dominance under fixed rival bids with one and two rivals.
2. **Part 2 (First-Price Sealed-Bid Auction):** Evaluates strategic deviations where Bidder 1 bids $b_1 = k \cdot v_1$ for $k \in [0.1, 2.0]$ against $n-1$ rivals adhering to the Bayesian Nash Equilibrium (BNE) $b_i = \frac{n-1}{n} v_i$ for $n \in \{2, 3, 4\}$. Empirically confirms that the payoff-maximizing deviation factor coincides with $k^* = \frac{n-1}{n}$ and proves that the winner overpays in 100% of continuous-bid auctions.
3. **Part 3 (VCG Mechanism for Shortest-Path Procurement):** Implements network path procurement on the classic benchmark from Shoham and Leyton-Brown (Figure 7.3). Reproduces the optimal path $A \to B \to E \to F$ (cost 4) and exact Clarke pivot payments ($p_{AB}=4, p_{BE}=2, p_{EF}=3$), and conducts misreporting experiments across $c' \in [0.0, 10.0]$ (step 0.5) proving that no edge agent can strictly benefit by lying.

---


## 2. Theoretical Foundations

### 2.1 Second-Price (Vickrey) Sealed-Bid Auction

In a single-item second-price auction, each agent $i$ submits a sealed bid $b_i$. The item is awarded to the highest bidder:

$$
i^{*} = \arg\max_{i} b_i
$$

The winner pays the **second-highest bid**:

$$
p_i^* = \max_j \neq i^* b_j
$$

All losing bidders pay zero. The utility of agent $i$ with true private value $v_i$ is:

$$
u_i(b_i, \mathbf{b}_{-i}) =
\begin{cases}
v_i - \max_{j \neq i} b_j & \text{if } b_i > \max_{j \neq i} b_j \\
0 & \text{if } b_i < \max_{j \neq i} b_j
\end{cases}
$$

**Dominant-Strategy Incentive Compatibility (DSIC):**

Bidding truthfully ($b_i = v_i$) is a weakly dominant strategy:

- If an agent underbids ($b_i < v_i$), they never reduce the price paid upon winning (which is determined solely by rival bids), but they risk losing when $\max_{j \neq i} b_j \in (b_i, v_i)$, forfeiting positive utility $v_i - \max_{j \neq i} b_j > 0$.
- If an agent overbids ($b_i > v_i$), they win only when they would have won anyway (at the exact same price), or they win when $\max_{j \neq i} b_j \in (v_i, b_i)$, resulting in negative utility $v_i - \max_{j \neq i} b_j < 0$.
- Hence, $u_i(v_i, \mathbf{b}_{-i}) \ge u_i(b_i', \mathbf{b}_{-i})$ for all $b_i'$ and all rival bids $\mathbf{b}_{-i}$.

### 2.2 First-Price Sealed-Bid Auction & Bayesian Nash Equilibrium

In a first-price auction, the winner pays their **own bid**:

$$
p_i^* = b_i^*
$$

$$
u_i =
\begin{cases}
v_i - b_i & \text{if } b_i > \max_{j \neq i} b_j \\
0 & \text{if } b_i < \max_{j \neq i} b_j
\end{cases}
$$

Because paying one's own bid yields zero surplus when bidding truthfully ($v_i - v_i = 0$), rational bidders shade their bids downward to trade off win probability against profit margin.

For $n$ symmetric risk-neutral bidders with valuations $v_i \sim U[0, 1]$ i.i.d., the unique symmetric Bayesian Nash Equilibrium (BNE) strategy is:

$$
b^{*}(v_i) = \left( \frac{n - 1}{n} \right) v_i
$$

**Equilibrium Deviation Analysis:**

When Bidder 1 bids $b_1 = k \cdot v_1$ while the remaining $n-1$ rivals bid $b^{*}(v_j) = \frac{n-1}{n} v_j$, the expected utility of Bidder 1 for $k \le \frac{n-1}{n}$ is:

$$
\mathbb{E}[u_1(k)] =
\frac{1-k}{n+1}
\left( \frac{nk}{n-1} \right)^{n-1}
$$

Differentiating with respect to $k$ yields:

$$
\frac{d}{dk} \left[ (1-k)k^{n-1} \right]
= k^{n-2} \left[ (n-1)(1-k)-k \right] = 0
$$

$$
\implies k^{*} = \frac{n-1}{n}
$$

At this equilibrium, the expected utility achieved is:

$$
\mathbb{E}[u_1(k^{*})] = \frac{1}{n(n+1)}
$$

**Structural Winner's Regret:**

Because bid values are continuous real numbers, the event of an exact tie has probability zero ($P(b_{(1)} = b_{(2)}) = 0$). Consequently, in **100% of winning auctions**, the winner pays strictly more than the second-highest bid ($b_{(1)} > b_{(2)}$). The winner could always ex-post have bid $b_{(2)} + \epsilon$ and still won. The expected gap $\mathbb{E}[b_{(1)} - b_{(2)}]$ represents the average surplus left on the table.

### 2.3 The Vickrey-Clarke-Groves (VCG) Mechanism

In network path procurement, a buyer seeks the cheapest path from source $A$ to destination $F$ across a graph $G = (V, E)$, where each edge $e$ is an autonomous agent with private cost $c_e$.

The mechanism selects the path $P^{*}$ minimizing total reported cost:

$$
P^{*} = \arg\min_{P \in \mathcal{P}_{A \to F}} \sum_{e \in P} c_e
$$

Under the Clarke pivot rule, each selected edge $e \in P^{*}$ receives a payment equal to its reported cost plus the externality it imposes on the remaining network:

$$
p_e = d(G \setminus \{e\}) - \left( d(G) - c_e \right)
$$

where $d(G)$ is the shortest path cost in $G$, and $d(G \setminus \{e\})$ is the shortest path cost when edge $e$ is unavailable. Unselected edges ($e \notin P^{*}$) receive $p_e = 0$.

Because unselected edges do not route traffic, they incur no cost. The utility is:

$$
u_e =
\begin{cases}
p_e - c_e^{\text{true}} = d(G \setminus \{e\}) - d(G) & \text{if } e \in P^* \\
0 & \text{if } e \notin P^*
\end{cases}
$$

VCG guarantees:

1. **Strategyproofness (Truthfulness):** Truthful reporting ($c_e' = c_e^{\text{true}}$) is a dominant strategy for all agents.
2. **Individual Rationality:** $u_e \ge 0$ under truthful reporting; no participating agent suffers a financial loss.

---

## 3. Repository Structure

```
├── part1_second_price_auction.py       # Part 1: Second-price auction simulation & verification
├── part2_first_price_auction.py        # Part 2: First-price auction equilibrium & overpayment
├── part3_vcg_mechanism.py              # Part 3: VCG cheapest path procurement mechanism
├── requirements.txt                    # Project Python package dependencies
├── README.md                           # Comprehensive documentation and project guide
└── plots/                              # High-resolution (300 DPI) publication figures
    ├── part1a_one_rival.png            # Part 1A: 3x3 grid (1 rival, 3 strategies)
    ├── part1b_fixed_rival.png          # Part 1B: 3x3 grid (fixed rival bids)
    ├── part1c_two_rivals.png           # Part 1C: 3x3 grid (2 rivals, stochastic)
    ├── part2_utility.png               # Part 2: Expected utility vs k (n=2, 3, 4)
    ├── part2_overpaid.png              # Part 2: Overpayment fraction and regret margin
    └── part3_misreporting.png          # Part 3: Strategic misreporting 2x2 grid
```

---

## 4. Installation & Requirements

Ensure Python 3.10+ is installed. Install all necessary dependencies via `pip`:

```bash
pip install -r requirements.txt
```

*Contents of `requirements.txt`:*
```
numpy
matplotlib
networkx
```

---

## 5. Execution Instructions

Each component of the assignment can be executed independently from the terminal:

### Run Part 1 (Second-Price Auction Simulation):
```bash
python part1_second_price_auction.py
```
*Outputs:* Simulates $N = 100{,}000$ auctions per bid point, performs analytical dominance checks, and saves `part1a_one_rival.png`, `part1b_fixed_rival.png`, and `part1c_two_rivals.png` to `/plots`.

### Run Part 2 (First-Price Auction Simulation):
```bash
python part2_first_price_auction.py
```
*Outputs:* Evaluates deviation factor $k \in [0.1, 2.0]$ across $n \in \{2, 3, 4\}$, verifies the Bayesian Nash Equilibrium peak, measures overpayment frequency and regret margins, and saves `part2_utility.png` and `part2_overpaid.png` to `/plots`.

### Run Part 3 (VCG Mechanism):
```bash
python part3_vcg_mechanism.py
```
*Outputs:* Computes optimal path $A \to B \to E \to F$, reproduces exact textbook payments ($p_{AB}=4, p_{BE}=2, p_{EF}=3$), and evaluates strategic deviations across $[0.0, 10.0]$ saved in `part3_misreporting.png`.

---

## 6. Detailed Experimental Results & Analysis

### Part 1: Second-Price Sealed-Bid Auction

#### Part 1A: One Rival Bidder under Three Strategies
We evaluated Bidder 1 across valuations $v_1 \in \{0.2, 0.5, 0.8\}$ against three rival strategies:
1. **Truthful:** $b_2 = v_2$
2. **Half-Value:** $b_2 = v_2 / 2$
3. **Random Uniform:** $b_2 \sim U[0, 1]$

| $v_1$ | Rival Strategy | Truthful Utility | Simulated Peak Utility | Peak Bid $b_1$ | Optimal Range (Plateau) |
|:---:|:---|:---:|:---:|:---:|:---:|
| **0.2** | Truthful ($b_2 = v_2$) | 0.0200 | 0.0200 | 0.20 | [0.19, 0.21] |
| **0.2** | Half-Value ($b_2 = v_2/2$) | 0.0398 | 0.0398 | 0.20 | [0.19, 0.21] |
| **0.2** | Random Uniform | 0.0200 | 0.0200 | 0.20 | [0.19, 0.21] |
| **0.5** | Truthful ($b_2 = v_2$) | 0.1243 | 0.1243 | 0.50 | [0.49, 0.51] |
| **0.5** | Half-Value ($b_2 = v_2/2$) | 0.2502 | 0.2502 | 0.50 | **[0.50, 1.00]** |
| **0.5** | Random Uniform | 0.1249 | 0.1249 | 0.50 | [0.49, 0.51] |
| **0.8** | Truthful ($b_2 = v_2$) | 0.3181 | 0.3181 | 0.80 | [0.79, 0.81] |
| **0.8** | Half-Value ($b_2 = v_2/2$) | 0.5501 | 0.5501 | 0.50 | **[0.50, 1.00]** |
| **0.8** | Random Uniform | 0.3206 | 0.3206 | 0.80 | [0.79, 0.81] |

**Do these plots help Bidder 1 select its strategy?**
**Yes, unequivocally.** The plots establish that bidding truthfully ($b_1 = v_1$) is weakly dominant:
- Under truthful and random rival strategies, the utility curve has a unique strict maximum at $b_1 = v_1$.
- Under the half-value rival strategy, the rival bids at most $0.5$. For $v_1 = 0.8$, any bid $b_1 \ge 0.5$ wins with certainty, forming an optimal utility plateau over $[0.50, 1.00]$. The truthful bid $b_1 = 0.8$ lies directly on this optimal plateau.
- Most importantly, Bidder 1 **does not need to know or predict the rival's strategy**; bidding $b_1 = v_1$ is guaranteed to be optimal across all possible rival behaviors.

#### Part 1B: Fixed Rival Bid Analytical Verification & General Dominance Proof

##### 1. Empirical Verification on Fixed Grid ($b_2 \in \{0.1, 0.4, 0.9\}$)
To verify the mechanism under deterministic rival actions, we fixed rival bids at $b_2 \in \{0.1, 0.4, 0.9\}$ across all $v_1 \in \{0.2, 0.5, 0.8\}$, yielding 9 exact configurations:

| $v_1$ | Rival Bid $b_2$ | Condition | Truthful Utility $u(v_1)$ | Max Utility $\max_{b_1} u(b_1)$ | Winning Optimal Bid Range | Truthful Optimal? |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **0.2** | 0.1 | $v_1 > b_2$ (Profitable to win) | **0.10** | 0.10 | $b_1 \in (0.10, 1.00]$ | **Yes (Dominant)** |
| **0.2** | 0.4 | $v_1 < b_2$ (Loss to win) | **0.00** | 0.00 | $b_1 \in [0.00, 0.40)$ | **Yes (Dominant)** |
| **0.2** | 0.9 | $v_1 < b_2$ (Loss to win) | **0.00** | 0.00 | $b_1 \in [0.00, 0.90)$ | **Yes (Dominant)** |
| **0.5** | 0.1 | $v_1 > b_2$ (Profitable to win) | **0.40** | 0.40 | $b_1 \in (0.10, 1.00]$ | **Yes (Dominant)** |
| **0.5** | 0.4 | $v_1 > b_2$ (Profitable to win) | **0.10** | 0.10 | $b_1 \in (0.40, 1.00]$ | **Yes (Dominant)** |
| **0.5** | 0.9 | $v_1 < b_2$ (Loss to win) | **0.00** | 0.00 | $b_1 \in [0.00, 0.90)$ | **Yes (Dominant)** |
| **0.8** | 0.1 | $v_1 > b_2$ (Profitable to win) | **0.70** | 0.70 | $b_1 \in (0.10, 1.00]$ | **Yes (Dominant)** |
| **0.8** | 0.4 | $v_1 > b_2$ (Profitable to win) | **0.40** | 0.40 | $b_1 \in (0.40, 1.00]$ | **Yes (Dominant)** |
| **0.8** | 0.9 | $v_1 < b_2$ (Loss to win) | **0.00** | 0.00 | $b_1 \in [0.00, 0.90)$ | **Yes (Dominant)** |

In all 9 cases, bidding $b_1 = v_1$ yields the highest possible utility ($u(v_1) = \max_{b_1} u(b_1)$).

##### 2. General Theoretical Dominance Proof for Every Rival Bid ($b_2 \in [0, \infty)$)
While the 9-point grid verifies specific test cases, game theory establishes weak dominance across **all possible rival bids** $b_2 \in [0, \infty)$ via three exhaustive cases:
1. **Case 1: $b_2 < v_1$ (Winning is profitable):**
   - Truthful bid $b_1 = v_1 > b_2$: Bidder 1 wins and pays $b_2$, achieving strictly positive utility $u(v_1) = v_1 - b_2 > 0$.
   - Any other bid $b_1' > b_2$ also wins and pays $b_2$, yielding identical utility $v_1 - b_2$.
   - Any bid $b_1' < b_2$ loses, yielding utility $0 < v_1 - b_2$.
   - Hence, $u(v_1) \ge u(b_1')$ for all $b_1'$.
2. **Case 2: $b_2 > v_1$ (Winning causes a loss):**
   - Truthful bid $b_1 = v_1 < b_2$: Bidder 1 loses, achieving utility $u(v_1) = 0$.
   - Any other losing bid $b_1' < b_2$ yields utility $0$.
   - Any winning bid $b_1' > b_2$ forces Bidder 1 to pay $b_2$, resulting in negative utility $u(b_1') = v_1 - b_2 < 0$.
   - Hence, $u(v_1) \ge u(b_1')$ for all $b_1'$.
3. **Case 3: $b_2 = v_1$ (Breakeven):**
   - Any bid $b_1'$ yields utility $0$ (either loses at payoff 0, or wins at price $b_2 = v_1$ with payoff $v_1 - v_1 = 0$).
   - Hence, $u(v_1) = u(b_1') = 0$.

Therefore, truthful bidding $b_1 = v_1$ is unconditionally weakly dominant for every rival bid $b_2 \in [0, \infty)$.

#### Part 1C: Two Rival Bidders (Repeating Parts A and B with Two Rivals)

##### 1. Stochastic Rival Strategies (Part A with Two Rivals)
With two rival bidders (Bidders 2 and 3) adopting identical strategies, Bidder 1 wins if and only if $b_1 > \max(b_2, b_3)$, paying the second price $p = \max(b_2, b_3)$:
- **Truthful Rivals ($b_2 = v_2, b_3 = v_3$):** Second price distribution is $F_{\max}(x) = x^2$. Truthful bidding $b_1 = v_1$ achieves peak utility (0.0027 for $v_1=0.2$, 0.0417 for $v_1=0.5$, 0.1707 for $v_1=0.8$).
- **Half-Value Rivals ($b_2 = v_2/2, b_3 = v_3/2$):** Rivals bid at most $0.5$. For $v_1 = 0.8$, Bidder 1 achieves a maximal utility plateau over $[0.50, 1.00]$ ($u \approx 0.3833$), where $b_1 = 0.8$ lies comfortably.
- **Random Rivals ($b_2, b_3 \sim U[0, 1]$):** Peak utility strictly aligns with $b_1 = v_1$.

##### 2. Fixed Rival Bid Verification (Part B with Two Rivals)
To repeat Part B with two rivals, we evaluate Bidder 1's utility against all pairs $(b_2, b_3) \in \{0.1, 0.4, 0.9\}^2$. In a second-price auction, two rivals exert an effective rival bid $p_{\text{rival}} = \max(b_2, b_3)$. Evaluating all 27 combinations ($3$ values of $v_1 \times 9$ pairs of $(b_2, b_3)$):

| $v_1$ | Rival Pair $(b_2, b_3)$ | Effective Price $\max(b_2, b_3)$ | Truthful Utility $u(v_1)$ | Max Utility $\max_{b_1} u(b_1)$ | Truthful Optimal? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.2** | $(0.1, 0.1)$ | 0.1 | **0.10** | 0.10 | **Yes** |
| **0.2** | $(0.1, 0.4)$ or $(0.4, 0.1)$ or $(0.4, 0.4)$ | 0.4 | **0.00** | 0.00 | **Yes** |
| **0.2** | Any pair with 0.9 | 0.9 | **0.00** | 0.00 | **Yes** |
| **0.5** | $(0.1, 0.1)$ | 0.1 | **0.40** | 0.40 | **Yes** |
| **0.5** | $(0.1, 0.4)$ or $(0.4, 0.1)$ or $(0.4, 0.4)$ | 0.4 | **0.10** | 0.10 | **Yes** |
| **0.5** | Any pair with 0.9 | 0.9 | **0.00** | 0.00 | **Yes** |
| **0.8** | $(0.1, 0.1)$ | 0.1 | **0.70** | 0.70 | **Yes** |
| **0.8** | $(0.1, 0.4)$ or $(0.4, 0.1)$ or $(0.4, 0.4)$ | 0.4 | **0.40** | 0.40 | **Yes** |
| **0.8** | Any pair with 0.9 | 0.9 | **0.00** | 0.00 | **Yes** |

In all 27 combinations, $u(v_1) = \max_{b_1} u(b_1)$. Truthful bidding $b_1 = v_1$ is verified to be weakly dominant against any pair of rival bids.

---

### Part 2: First-Price Sealed-Bid Auction

Bidder 1 deviated by bidding $b_1 = k \cdot v_1$ for $k \in [0.1, 2.0]$, while $n-1$ rivals played equilibrium $b^*(v_i) = \frac{n-1}{n} v_i$.

| $n$ | Theoretical $k^*$ | Sim Peak $k$ | Theoretical $\mathbb{E}[U]$ | Sim Peak $\mathbb{E}[U]$ | Win % @ BNE | Winner Regret % | Regret Margin |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2** | 0.500 | 0.500 | 0.16667 ($1/6$) | 0.16600 | 49.8% | **100.0%** | 0.1662 |
| **3** | 0.667 | 0.650 | 0.08333 ($1/12$) | 0.08328 | 33.1% | **100.0%** | 0.1671 |
| **4** | 0.750 | 0.750 | 0.05000 ($1/20$) | 0.05001 | 25.0% | **100.0%** | 0.1504 |

**Overpayment Analysis ("Could the winner have paid less and still won?"):**
1. **100.0% Frequency (Continuous Bids Without Ties):** Across all $100{,}000$ auctions, in **100.0% of winning auctions**, the winner paid strictly more than the second-highest bid ($b_{(1)} > b_{(2)}$). Since bids are continuous random variables drawn from uniform distributions, exact ties have probability zero ($P(b_{(1)} = b_{(2)}) = 0$). The winner could always ex-post have bid $b_{(2)} + \epsilon$ and still won.
2. **Crucial Distinction — Overpayment vs. Financial Loss:**
   > **Note on Terminology:** "Overpaying" in auction theory refers strictly to *ex-post regret*—paying more than the minimum necessary bid to secure the item, thereby leaving surplus on the table. **It does NOT mean the winner suffers a financial loss.**
   > 
   > In fact, at the BNE equilibrium $k^* = \frac{n-1}{n} < 1$, the winner's payoff is:
   > $$u_1 = v_1 - b_1 = v_1 \left(1 - \frac{n-1}{n}\right) = \frac{v_1}{n} > 0 \quad (\forall v_1 > 0)$$
   > The winner always earns a strictly positive profit. Overpayment is purely an informational externality of the first-price rule compared to the second-price rule.
3. **Surplus Left on Table:** The average overpayment margin $\mathbb{E}[b_{(1)} - b_{(2)}]$ equals $0.1662$ for $n=2$, $0.1671$ for $n=3$, and decreases to $0.1504$ for $n=4$. As $n$ grows, order statistics crowd closer, reducing the overpayment gap.

---

### Part 3: VCG Mechanism for Shortest-Path Procurement

Using Figure 7.3 from Shoham and Leyton-Brown:
- Nodes: $\{A, B, C, D, E, F\}$
- True edge costs: $AB=1, AC=4, BD=1, BE=2, CD=1, DE=1, DF=4, EF=1$.

#### 1. Baseline Shortest-Path Tie-Breaking (ABEF vs. ABDEF)
In the baseline network, there is an exact cost tie between two paths from $A$ to $F$:
- **Path $ABEF$:** Cost $= 1(AB) + 2(BE) + 1(EF) = 4.0$ (3 edges / 4 nodes)
- **Path $ABDEF$:** Cost $= 1(AB) + 1(BD) + 1(DE) + 1(EF) = 4.0$ (4 edges / 5 nodes)

To reproduce the textbook payments ($p_{AB}=4, p_{BE}=2, p_{EF}=3$), the mechanism **must select path $ABEF$**. If $ABDEF$ were selected instead, edge $BE$ would receive payment $0$, violating the textbook result.

**Deterministic Tie-Breaking Rule Implemented:**
Our implementation enforces a consistent, deterministic tie-breaking hierarchy:
1. *Primary:* Minimal total reported path cost
2. *Secondary:* Minimal hop-count (fewest edges / nodes)
3. *Tertiary:* Lexicographical node sequence

Because path $ABEF$ has 3 hops while path $ABDEF$ has 4 hops, $ABEF$ is selected deterministically and unconditionally.

#### 2. VCG Payment & Utility Results

| Edge $(u, v)$ | True Cost $c_e$ | Reported Cost | Selected on Path? | VCG Payment $p_e$ | Utility $u_e = p_e - c_e$ | Textbook Benchmark |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **(A, B)** | 1.0 | 1.0 | **YES** | **4.00** | 3.00 | **4.0 (Matched ✓)** |
| **(A, C)** | 4.0 | 4.0 | NO | 0.00 | 0.00 | **0.0 (Matched ✓)** |
| **(B, D)** | 1.0 | 1.0 | NO | 0.00 | 0.00 | **0.0 (Matched ✓)** |
| **(B, E)** | 2.0 | 2.0 | **YES** | **2.00** | 0.00 | **2.0 (Matched ✓)** |
| **(C, D)** | 1.0 | 1.0 | NO | 0.00 | 0.00 | **0.0 (Matched ✓)** |
| **(D, E)** | 1.0 | 1.0 | NO | 0.00 | 0.00 | **0.0 (Matched ✓)** |
| **(D, F)** | 4.0 | 4.0 | NO | 0.00 | 0.00 | **0.0 (Matched ✓)** |
| **(E, F)** | 1.0 | 1.0 | **YES** | **3.00** | 2.00 | **3.0 (Matched ✓)** |
| **TOTAL** | - | - | - | **9.00** | **5.00** | **All Verified ✓** |

#### 3. Strategic Misreporting Analysis & Tie at $c'(AC) = 1.0$
We evaluated reported costs from $0.0$ to $10.0$ in steps of $0.5$:
- **Winning Edge $AB$ (True cost 1.0):** For $c' \in [0.0, 4.0]$, $AB$ remains on path, receiving payment $4.0$ and utility $3.0$. For $c' > 4.0$, path $ACDEF$ (cost 7.0) becomes cheaper; $AB$ is excluded, receiving payment $0.0$ and utility $0.0$. No misreport can exceed utility $3.0$.
- **Losing Edge $AC$ (True cost 4.0):**
  - For $c' > 1.0$: Path $ACDEF$ has cost $> 4.0$; $AC$ remains excluded, earning utility $0.0$.
  - **At Boundary $c'(AC) = 1.0$ (Tie Condition):**
    Path $ACDEF$ has total cost $1.0(AC) + 1(CD) + 1(DE) + 1(EF) = 4.0$, tying with path $ABEF$ (cost 4.0).
    - Under our minimal hop-count tie-breaking rule (3 hops for $ABEF$ vs. 4 hops for $ACDEF$), $ABEF$ wins, so $AC$ is excluded ($u_{AC} = 0.0$).
    - If an alternative tie-breaker selected $ACDEF$, $AC$'s VCG payment would be:
      $$p_{AC} = d(G \setminus \{AC\}) - [d(G) - c'(AC)] = 4.0 - (4.0 - 1.0) = 1.0$$
      Because $AC$'s true cost is $4.0$, its utility would be:
      $$u_{AC} = p_{AC} - c_{AC}^{\text{true}} = 1.0 - 4.0 = -3.0 < 0 \quad (\text{Severe loss of \$3.00})$$
  - For underreporting $c' < 1.0$: Path $ACDEF$ has cost $c' + 3 < 4.0$, entering the optimal path. However, its payment is strictly capped at $1.0$, resulting in negative utility $u_{AC} = 1.0 - 4.0 = -3.0 < 0$.
  - **Conclusion:** Under every possible tie-breaking rule, edge $AC$ achieves its maximum utility ($0.0$) by reporting truthfully. Lying to win induces a devastating financial loss.

---

## 7. Computational Complexity & Scalability Analysis

| Mechanism Component | Dominant Operation | Time Complexity | Typical Runtime |
|:---|:---|:---:|:---:|
| **Part 1 (Second-Price Simulation)** | Vectorized broadcasting over $N$ auctions | $O(\|B\| \cdot N)$ | $\approx 1.2$ s |
| **Part 2 (First-Price Simulation)** | Row-wise sorting of $N \times n$ bid matrix | $O(\|K\| \cdot N \cdot n \log n)$ | $\approx 2.5$ s |
| **Part 3 (VCG Mechanism)** | Dijkstra shortest path on $G$ and $G \setminus \{e\}$ | $O(\|P^*\| \cdot (\|E\| + \|V\| \log \|V\|))$ | $\approx 0.05$ s |

All algorithms scale polynomially, enabling straightforward extension to large networks and high bidder populations.

---

## 8. Limitations & Practical Considerations

1. **Collusion Vulnerability:** VCG is strategyproof against unilateral deviations, but vulnerable to agent coalitions. For example, if edges $AB$ and $AC$ colluded, $AC$ could report cost $100$, driving $AB$'s payment to $100$ and extracting unbounded payments from the buyer.
2. **High Total Buyer Expenditure:** In Part 3, the buyer pays $\$9.00$ for a path that costs only $\$4.00$ (a 125% premium). Bottleneck links with high market power extract significant information rents.
3. **NP-Hardness in Combinatorial Allocations:** In general combinatorial auctions with arbitrary package valuations, the winner determination problem is NP-hard, requiring approximation algorithms where exact VCG truthfulness may be sacrificed.

---

## 9. Conclusions

1. **Second-Price Auctions:** Truthful bidding is weakly dominant across all rival strategies and valuations, with empirical utility peaking at $b_1 = v_1$.
2. **First-Price Auctions:** Expected utility peaks at the Bayesian Nash Equilibrium $k^* = (n-1)/n$ across $n \in \{2, 3, 4\}$. In 100% of auctions, the winner overpays relative to the minimum winning bid, demonstrating structural regret.
3. **VCG Mechanism:** Textbook payments ($p_{AB}=4, p_{BE}=2, p_{EF}=3$) were exactly reproduced. Strategic deviations proved that neither winning nor losing edges can profit by lying, validating strategyproofness and individual rationality.

---

## 10. References

1. Y. Shoham and K. Leyton-Brown, *Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations*, Cambridge University Press, 2008.
2. W. Vickrey, "Counterspeculation, auctions, and competitive sealed tenders," *Journal of Finance*, vol. 16, no. 1, pp. 8–37, 1961.
3. E. H. Clarke, "Multipart pricing of public goods," *Public Choice*, vol. 11, pp. 17–33, 1971.
4. T. Groves, "Incentives in teams," *Econometrica*, vol. 41, no. 4, pp. 617–631, 1973.
5. V. Krishna, *Auction Theory*, Academic Press, 2nd ed., 2009.
6. N. Nisan, T. Roughgarden, E. Tardos, and V. Vazirani, *Algorithmic Game Theory*, Cambridge University Press, 2007.
