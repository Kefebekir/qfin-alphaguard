"""How the shadow record's betting test behaves (evidence.py; decision 0020), by simulation.

Daily differences between a challenger at equal risk and the rule, normal with a daily
standard deviation of 1%, clipped at 5%, in records of 12 years:

1. with no edge, the share of 2,000 records ever promoted at alpha 5% and 2.5% (Ville's
   inequality bounds it by alpha), and, for comparison, the share whose t-statistic, read
   every evening from the 20th day, ever reaches 1.96 (a one-sided 2.5% test read once);
2. with an edge whose annual Sharpe ratio is 0.5, 1, 2 or 3, the share of 2,000 records
   promoted within 2, 4 and 8 years, and the median time, for a card whose share of the total
   alpha (2.5%) is all of it, half, or a fifth;
3. five challengers with no edge, added a year apart, in 20,000 runs: the share of runs in
   which any is promoted, when each card's share is fixed at alpha / 5 (decision 0020), and
   when the threshold is instead the number of cards so far over alpha;
4. the clip: with an edge of Sharpe 2 and a daily difference whose standard deviation is
   0.25% to 2%, the median time to promotion at alpha 2.5% (1,000 records) with a clip of 5%,
   and with the clip from 10 training years of the same kind (evidence.training_settings: the
   largest difference, rounded up to 0.1%);
5. an insurance seller, no better on average: +0.04% a day, and -9.96% on a day with chance
   1 / 250. The share of 2,000 records promoted in 12 years at clips of 5% and 10%.
"""

import math

import numpy as np

from qfin_alphaguard.evidence import betting_wealth, round_up

YEAR = 252
rng = np.random.default_rng(7)
paths, days, sd = 2000, YEAR * 12, 0.01


def outcomes(mean, n_paths, n_days):
    return np.clip(rng.normal(mean, sd, (n_paths, n_days)), -0.05, 0.05) / 0.05


x = outcomes(0.0, paths, days)
_, w = betting_wealth(x)
print("1. no edge, share ever promoted in 12 years:")
print(
    f"   alpha 5%: {(w.max(1) >= 20).mean():.4f}, alpha 2.5%: {(w.max(1) >= 40).mean():.4f}"
)
n = np.arange(1, days + 1)
mean = np.cumsum(x, 1) / n
variance = (np.cumsum(x**2, 1) - n * mean**2) / np.maximum(n - 1, 1)
t = mean[:, 19:] / np.sqrt(variance[:, 19:] / n[19:])  # from the 20th day
print(f"   a t-test read every evening, t >= 1.96: {(t >= 1.96).any(1).mean():.4f}")

print("2. annual Sharpe of the difference: promoted within 2 / 4 / 8 years, median")
for sharpe in (0.5, 1.0, 2.0, 3.0):
    _, w = betting_wealth(outcomes(sharpe / math.sqrt(YEAR) * sd, paths, days))
    for alpha in (0.025, 0.0125, 0.005):
        hit = w >= 1 / alpha
        first = np.where(hit.any(1), hit.argmax(1), np.inf) / YEAR
        median = np.median(first)
        print(
            f"   Sharpe {sharpe}, card alpha {alpha:.2%} (threshold {1 / alpha:.0f}): "
            f"{np.mean(first <= 2):.0%} / {np.mean(first <= 4):.0%} / "
            f"{np.mean(first <= 8):.0%}, median "
            + (f"{median:.1f} years" if median < np.inf else "over 12 years")
        )

runs, cards, total = 20000, 5, 0.025
fixed, counted = np.zeros(runs, bool), np.zeros(runs, bool)
for k in range(cards):  # card k starts in year k and runs to the end
    n = days - YEAR * k
    _, w = betting_wealth(outcomes(0.0, runs, n))
    so_far = np.minimum((np.arange(n) + YEAR * k) // YEAR + 1, cards)
    fixed |= (w >= cards / total).any(1)
    counted |= (w >= so_far / total).any(1)
print(f"3. {cards} challengers with no edge, a year apart: share of runs promoting any")
print(f"   share fixed at alpha / {cards}: {fixed.mean():.4f}")
print(f"   threshold cards so far / alpha: {counted.mean():.4f}   (alpha {total})")

print(
    "4. Sharpe 2, alpha 2.5%: median years to promotion, clip 5% / clip from training"
)
for sd_d in (0.0025, 0.005, 0.01, 0.02):
    mean = 2 / math.sqrt(YEAR) * sd_d
    d = rng.normal(mean, sd_d, (1000, days))
    # each record's own clip, from its own 10 training years
    largest = np.abs(rng.normal(mean, sd_d, (1000, YEAR * 10))).max(1)
    trained = np.array([round_up(float(v)) for v in largest])[:, None]
    row = []
    for clip in (0.05, trained):
        _, w = betting_wealth(np.clip(d, -clip, clip) / clip)
        hit = w >= 40
        median = np.median(np.where(hit.any(1), hit.argmax(1), np.inf) / YEAR)
        row.append(f"{median:.1f}" if median < np.inf else "over 12")
    print(
        f"   daily sd {sd_d:.2%}: {row[0]} / {row[1]} "
        f"(median clip from training {np.median(trained):.1%})"
    )

print("5. insurance seller, no edge: share promoted in 12 years (alpha 2.5%)")
d = np.where(rng.random((2000, days)) < 1 / 250, -0.0996, 0.0004)
for clip in (0.05, 0.10):
    _, w = betting_wealth(np.clip(d, -clip, clip) / clip)
    print(f"   clip {clip:.0%}: {(w.max(1) >= 40).mean():.3f}")
