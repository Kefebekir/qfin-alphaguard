# Machine learning for QFin-AlphaGuard: the evidence, a pilot, and what to build

October 2026. This is the research behind decision 0017. It answers three
questions:
1. What can machine learning really do for a long-only US large-cap system?
2. Which approach is best for this project, with its data, costs and
   constraints?
3. How can a result be shown to be real, both before and after it trades
   forward?

Every number from a paper was checked in the paper itself, unless marked as
coming from a summary. "Our pilot" numbers come from `research/ml_pilot/`
and can be rerun.

## 1. The short answer

- **No model predicts stock returns with anything near 100% accuracy.** In
  the best-known study, the best models explain about 0.4% of the monthly
  variation in a stock's return out of sample. Edges are small, so the whole
  craft is in using them without fooling oneself and without paying them
  away in costs.
- **The best use of ML here is to decide which stocks the nightly plan
  holds:** a cross-sectional ranking of the coming month's (or week's)
  returns, from price, volume and fundamental features.
  - It should not time intraday trades; that is worth about 1 bp a trade.
  - It should not trade its own short-term signals; costs eat those.
- **The models to use** are a regularised linear model and gradient-boosted
  trees, averaged, with a small neural network ensemble as a challenger.
  - Deeper models enter only if they beat these under the same test.
  - LLMs need news data that is point-in-time, and they carry a look-ahead
    trap of their own (section 2.5).
- **Our own pilot finds a signal that is consistent but not yet proven.**
  - Setting: 2018–2026, 20 price and volume features.
  - Monthly gradient boosting picked 10 of the year's 50 most liquid stocks.
    It beat holding all 50 at equal weight by 9.3% a year after 10 bps of
    costs, in 8 of 9 years.
  - Corrected for the twelve variants tried, its deflated Sharpe ratio is
    0.70 against the 0.95 needed. It could still be luck.
- **Proof comes forward.** Each night's predictions are committed publicly
  before the open, a frozen model is paper traded, and the minimum track
  record length tells how long that takes. A modest edge (information ratio
  0.5) needs about 11 years to prove at 95%; a strong one (1.0), about 3.
  Paper trading first proves that the system trades as the backtest says.

## 2. What the literature shows

### 2.1 Predictability is real but small, and mostly where it is hard to trade

| Study | What it found |
| --- | --- |
| Gu, Kelly and Xiu (2020), *Review of Financial Studies* | US stocks 1957–2016, 94 characteristics. Out-of-sample monthly R² is 0.33–0.40% for trees and neural networks (NN3 best, 0.40%), 0.11–0.27% for regularised linear models, and −3.46% for plain OLS on all predictors. Among the 1,000 largest stocks: 0.52–0.70%. A value-weighted long-short decile portfolio from neural network forecasts has a Sharpe ratio of 1.35, against 0.61 for a leading linear model. The strongest predictors are recent price trends (short-term reversal, momentum, industry momentum), then liquidity and volatility. |
| Avramov, Cheng and Metzker (2023), *Management Science* | ML profits come largely from hard-to-arbitrage stocks. Excluding microcaps cuts the payoff by about 62%, excluding non-rated firms by about 68%, and they shrink further after reasonable trading costs because of high turnover. (Percentages from a summary of the paper.) |
| Jensen, Kelly, Malamud and Pedersen (2023 version; forthcoming, *Review of Financial Studies*) | A cost-agnostic ML Markowitz portfolio has a gross Sharpe of about 2.0, but turns negative after trading costs: it leans on fleeting small-stock signals such as one-month reversal. Learning portfolio weights with costs in the objective beats a tuned two-stage alternative by about 20% in Sharpe ratio and 60% in utility. After costs, value and quality become the most important features. Sample 1981–2020. |
| Cakici, Fieberg, Metko and Zaremba (2023, SSRN 4552395) | US stocks 1972–2020, 153 characteristics. ML works better for small firms, at the monthly rather than the yearly horizon, and in early data. For big firms, about 90% of US market value, yearly forecasts earned no significant abnormal return from 2001 to 2020. (From a summary of the paper.) |
| McLean and Pontiff (2016), *Journal of Finance* | Across 97 published predictors, returns are 26% lower out of sample and 58% lower after publication. |
| Harvey, Liu and Zhu (2016), *Review of Financial Studies* | With hundreds of factors tested, a new one needs a t-statistic above about 3.0, not 2.0. |
| DeMiguel, Garlappi and Uppal (2009), *Review of Financial Studies* | None of 14 optimised portfolio rules consistently beats equal weighting out of sample: estimation error eats the gain. Our min-variance plan losing to equal weight (decision 0015) is this result again. |

### 2.2 S&P 500 stocks at daily and weekly horizons

| Study | What it found |
| --- | --- |
| Krauss, Do and Huck (2017), *European Journal of Operational Research* | Deep nets, gradient-boosted trees and random forests, trained on lagged returns of S&P 500 members, 10 long and 10 short each day, 1992–2015. The ensemble earned 0.45% a day before costs and 0.25% after 5 bps per half-turn. The edge decayed: from January 2010 to October 2015 every model still earned 24–42% a year before costs, but lost 14–25% a year after them. |
| Jiang, Kelly and Xiu (2023), *Journal of Finance* | A CNN reading price-and-volume chart images predicts returns. The one-week long-short Sharpe ratio is 7.15 equal-weighted and 1.49 value-weighted. Restricted to the 500 largest stocks, it is still above 1.0. Net of 10–20 bps costs, the equal-weighted weekly strategy keeps a Sharpe ratio of up to 4.0, but monthly turnover is about 690%. |
| Chinco, Clark-Joseph and Ye (2019), *Journal of Finance* | LASSO on the whole cross-section's lagged returns predicts one-minute-ahead returns. The signals are sparse and short-lived. This is high-frequency work, out of reach for a retail broker's latency and spread. |

Lessons for us:
- The signal in large caps is real but small.
- Short-horizon signals are strong only before costs.
- Our cost, about 1 to 1.5 bps a side at Alpaca for small marketable orders
  (decision 0012), is several times lower than the 5–20 bps these studies
  assume. That moves the trade-off towards shorter horizons than an
  institution could use, but not to daily.

### 2.3 Long only

Blitz, Baltussen and van Vliet (2020, *Financial Analysts Journal*) find that
most of the added value of equity factor strategies comes from their long
legs. That supports a long-only design, the one our UK account, fractional
shares and Guard allow. (From the paper's abstract; the authors work at
Robeco and disclose it.)

### 2.4 Risk: volatility is far more predictable than returns

- **Christensen, Siggaard and Veliyev (2023, *Journal of Financial
  Econometrics*):** ML beats the HAR model in one-day-ahead realised variance
  of 29 Dow stocks, 2001–2017, even with HAR's own inputs. The gains grow
  with the horizon.
- **"HARd to Beat" (2024, arXiv 2406.08041):** HAR is as good when fitted on
  the right rolling window.
- **Volatility targeting:** Moreira and Muir's (2017, *Journal of Finance*)
  gains largely disappear out of sample in real time (Cederburg,
  O'Doherty, Wang and Yan, 2020, *Journal of Financial Economics*).

So: forecast risk with HAR on our 1-minute realised variance, try ML as a
challenger, and treat volatility targeting as a variant to test, not a
given.

### 2.5 Language models and news

- **Lopez-Lira and Tang (arXiv 2304.07619):** GPT-4 scores of news headlines
  predict next-day returns, more for small stocks and bad news. Returns
  fall as LLM use spreads.
- **The trap:** a model pretrained on text after the test period knows the
  future. Only headlines after the model's knowledge cutoff are clean.

We have no point-in-time news feed. This is not a Phase 2 candidate.

### 2.6 Execution timing

RL execution studies report gains of about 1 bp against TWAP or VWAP:
- +1.27 bps (a Stanford course project);
- 1.16 bps on Shanghai stocks (Hierarchical RL for VWAP, arXiv 2212.14670).

Our own measure (decision 0013) was 1.26 bps for 31 orders a stock a day.
At our turnover that is a few bps a year. The intraday model is therefore
the lowest priority; EqualSpaced stays.

### 2.7 How to test without fooling oneself

| Tool | Use |
| --- | --- |
| Purged walk-forward with an embargo (López de Prado, *Advances in Financial Machine Learning*, 2018) | No training label may overlap the test period. |
| Combinatorial purged cross-validation (CPCV) | Many test paths instead of one, for choosing models. |
| Deflated Sharpe ratio (Bailey and López de Prado, 2014, *Journal of Portfolio Management*) | Corrects the best Sharpe ratio for the number of trials and for non-normal returns. |
| Probability of backtest overfitting (Bailey, Borwein, López de Prado and Zhu, 2017, *Journal of Computational Finance*) | How often the in-sample winner is below the median out of sample. |
| Minimum track record length (Bailey and López de Prado, 2012, *Journal of Risk*) | How long a live record must be for its Sharpe ratio to mean anything. |
| t > 3 (Harvey, Liu and Zhu, 2016) | The bar for a new signal. |

## 3. Our pilot

Scripts and full output: `research/ml_pilot/`.

**Setup:**
- **Data:** every S&P 500 member on each day, 2016–2026: daily bars only, no
  fundamentals. About 500 stocks a day, 1.32 million stock-days.
- **20 features**, each ranked across stocks each day:
  - returns over 1, 5, 21, 63 and 126 days, and 12-1 momentum;
  - volatility over 21, 63 and 252 days, idiosyncratic volatility and beta
    against SPY;
  - the largest daily return, skewness, the high–low range, and overnight
    and intraday returns;
  - dollar volume, a volume shock, Amihud illiquidity, and the distance from
    the 52-week high.
- **Labels:** returns from the close of t+1 to t+1+h, h = 1, 5 or 21
  sessions.
- **Walk-forward by year:** test 2018 to October 2026, trained on everything
  before, purged of the last 1+h sessions.
- **Models:** ridge, gradient-boosted trees (scikit-learn's histogram GBT,
  LightGBM's algorithm) and their average, against three single signals.
  The hyperparameters were fixed before any test result was seen.
- **Portfolios:** long only, rebalanced every h sessions:
  - S&P 500: the top 10% of members against all members at equal weight;
  - our universe: the top 10 of the year's 50 most liquid against the 50 at
    equal weight.

**Rank IC** (correlation of each day's ranking with the coming returns; the
t-statistic uses non-overlapping periods):

| | Daily | Weekly | Monthly |
| --- | --- | --- | --- |
| Ridge | 0.009 (t 2.0) | 0.010 (t 1.4) | 0.012 (t 0.3) |
| Gradient-boosted trees | 0.007 (t 2.5) | 0.011 (t 1.7) | 0.017 (t 1.4) |
| Average of both | 0.009 (t 2.3) | 0.012 (t 1.5) | 0.016 (t 0.7) |
| 12-1 momentum | 0.017 (t 3.2) | 0.015 (t 1.2) | 0.005 (t 0.2) |

**Active return a year, long only, top 10 of our 50 against the 50:**

| | Gross | After 2 bps | After 10 bps | Turnover a year | Information ratio |
| --- | --- | --- | --- | --- | --- |
| Gradient-boosted trees, daily | +2.9% | −1.0% | −16.4% | 192× | 0.25 |
| Gradient-boosted trees, weekly | +8.1% | +7.2% | +3.7% | 44× | 0.65 |
| Gradient-boosted trees, monthly | +10.6% | +10.3% | +9.3% | 13× | 0.84 |
| Average of both, monthly | +8.9% | +8.6% | +7.4% | 15× | 0.83 |
| Ridge, monthly | +6.6% | +6.3% | +5.1% | 15× | 0.60 |
| 12-1 momentum, monthly | +4.6% | +4.5% | +3.9% | 7× | 0.30 |
| Low volatility, monthly | −7.0% | −7.1% | −7.5% | 6× | −0.51 |

**Monthly gradient-boosted trees, year by year:**

| 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| +4.3% | +0.5% | +16.2% | +20.1% | +7.8% | −1.0% | +10.5% | +22.1% | +10.9% |

It beat the 50 in 58% of months.

**What it means:**
- **The signal is small,** as in the literature: rank ICs of 0.01–0.02.
- **Daily trading loses to costs** even at our low cost; monthly and weekly
  holding keeps most of the gross.
- **The best result is not proven.** The deflated Sharpe ratio of the best
  of twelve variants is 0.53 weekly, 0.59 daily and 0.70 monthly, all below
  0.95, before counting the three horizons as trials too.
- **The fundamental law agrees.** With an IC near 0.016 over 50 stocks and
  12 rebalances a year, it implies an information ratio near 0.4
  (0.016 × √600). So the pilot's 0.84 probably includes luck.
- **2018–2026 was led by a few large growth stocks.** That flatters any
  model that leans towards them; low volatility lost about 7% a year.

## 4. What to build in Phase 2

1. **The data, point in time.**
   - Map the renamed tickers (the Phase 1 item), so no member's history is
     missing.
   - Add fundamentals from SEC EDGAR's XBRL company facts, which are free and
     carry the date each figure was filed, so a feature uses a figure only
     from its `filed` date. This gives value, profitability, investment and
     accruals: the features that survive costs (Jensen, Kelly, Malamud and
     Pedersen).
   - Industries from the SIC codes in EDGAR's submissions, for industry
     momentum and industry-neutral ranks.
   - Earnings dates from the filings.
   - EODHD's plan has no fundamentals (it answered 403 when tried).
2. **Targets.**
   - The cross-sectional rank of the coming 21-session return, the main
     one, and of the coming 5-session return.
   - Blended by alpha decay, as in Gârleanu and Pedersen (2013): aim in
     front of the target, trade part of the way. The band of decision 0006
     already does the second half.
3. **Models.**
   - Ridge, and gradient-boosted trees (LightGBM, which needs `libgomp1`),
     averaged by rank.
   - A challenger: an ensemble of small neural networks (Gu, Kelly and Xiu's
     NN3, several seeds).
   - Trained on every S&P 500 member, retrained each month, with each
     version registered.
4. **The portfolio.**
   - Long only, at most 25% a stock (Guard).
   - Candidates to compare, net of costs, at 500, 5,000 and 30,000 USD:
     1. top-N by forecast, equal weight;
     2. the forecast tilted on the min-variance plan;
     3. mean-variance with forecasts, Ledoit-Wolf risk and a trading-cost
        penalty.
   - Volatility targeting is a variant to test.
   - The universe traded: our 50 first. Wider only after 1-minute bars of
     the extra stocks are stored, and if the test says breadth pays.
5. **Execution:** EqualSpaced and Guard as they are. An ML timing model is a
   last, optional step.
6. **The test.**
   - Purged walk-forward for the result.
   - CPCV and nested tuning on training data only, for choices.
   - A registry of every variant tried, so the deflated Sharpe ratio counts
     them all.
   - The probability of backtest overfitting.
   - A new signal needs t > 3.
   - Benchmarks: SPY, the same stocks at equal weight, and the Phase 1 plan.
7. **The gate (M2).**
   - The ML plan's information ratio over equal weight has a deflated
     Sharpe ratio of at least 0.95.
   - It is positive after 10 bps in most years.
   - It beats SPY at similar risk.
   - If not, the honest result is "ML adds nothing here", and Phase 3 paper
     trades the Phase 1 plan.

## 5. Showing it forward

A backtest can be fitted to the past; a forward record cannot be, if no one
can change it after the fact.

1. **Freeze** a model version: code commit, training data range, and the
   hash of the weights.
2. **Commit each night's predictions and plan publicly before the open.**
   - A signed commit to the public repository, pushed before 09:30 New York
     time, with the hash of the predictions file.
   - Optionally anchored with OpenTimestamps.
   - Anyone can check that a prediction existed before the returns it
     predicts.
3. **Paper trade it on Alpaca** (Phase 3), and compare every day:
   - predicted ranks against realised returns (rank IC);
   - paper fills against the backtest's fills.
4. **Judge it by the numbers that hold forward:** the rank IC and active
   return, with confidence bands, against the backtest's expected range.

The minimum track record length (95%, one-sided, normal returns) says how
long it takes to show an active return is above zero:

| Information ratio | Months | Years |
| --- | --- | --- |
| 0.5 | 132 | 11.0 |
| 0.84 (the pilot's best) | 48 | 4.0 |
| 1.0 | 35 | 2.9 |

So the forward test's first job is to prove the system trades as the
backtest says (Phase 3's gate). Its second, over years, is to prove the
edge.

## References

- Avramov, D., Cheng, S., and Metzker, L. (2023). Machine Learning vs.
  Economic Restrictions: Evidence from Stock Return Predictability.
  *Management Science* 69(5), 2587–2619.
- Bailey, D. H., Borwein, J., López de Prado, M., and Zhu, Q. J. (2017). The
  Probability of Backtest Overfitting. *Journal of Computational Finance*
  20(4).
- Bailey, D. H., and López de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. *Journal of Risk* 15(2).
- Bailey, D. H., and López de Prado, M. (2014). The Deflated Sharpe Ratio.
  *Journal of Portfolio Management* 40(5), 94–107.
- Blitz, D., Baltussen, G., and van Vliet, P. (2020). When Equity Factors
  Drop Their Shorts. *Financial Analysts Journal* 76(4), 73–99.
- Cakici, N., Fieberg, C., Metko, D., and Zaremba, A. (2023). Predicting
  Returns with Machine Learning Across Horizons, Firm Size, and Time. SSRN
  4552395.
- Cederburg, S., O'Doherty, M. S., Wang, F., and Yan, X. (2020). On the
  Performance of Volatility-Managed Portfolios. *Journal of Financial
  Economics*.
- Chinco, A., Clark-Joseph, A. D., and Ye, M. (2019). Sparse Signals in the
  Cross-Section of Returns. *Journal of Finance* 74(1), 449–492.
- Christensen, K., Siggaard, M., and Veliyev, B. (2023). A Machine Learning
  Approach to Volatility Forecasting. *Journal of Financial Econometrics*
  21, 1680–1727.
- DeMiguel, V., Garlappi, L., and Uppal, R. (2009). Optimal Versus Naive
  Diversification. *Review of Financial Studies* 22(5), 1915–1953.
- Gârleanu, N., and Pedersen, L. H. (2013). Dynamic Trading with Predictable
  Returns and Transaction Costs. *Journal of Finance* 68(6), 2309–2340.
- Gu, S., Kelly, B., and Xiu, D. (2020). Empirical Asset Pricing via Machine
  Learning. *Review of Financial Studies* 33(5), 2223–2274.
- Harvey, C. R., Liu, Y., and Zhu, H. (2016). … and the Cross-Section of
  Expected Returns. *Review of Financial Studies* 29(1), 5–68.
- Jensen, T. I., Kelly, B., Malamud, S., and Pedersen, L. H. (2023 version).
  Machine Learning and the Implementable Efficient Frontier. Forthcoming,
  *Review of Financial Studies*. SSRN 4187217.
- Jiang, J., Kelly, B., and Xiu, D. (2023). (Re-)Imag(in)ing Price Trends.
  *Journal of Finance* 78(6), 3193–3249.
- Krauss, C., Do, X. A., and Huck, N. (2017). Deep Neural Networks,
  Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the
  S&P 500. *European Journal of Operational Research* 259(2), 689–702.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley.
- Lopez-Lira, A., and Tang, Y. Can ChatGPT Forecast Stock Price Movements?
  Return Predictability and Large Language Models. arXiv 2304.07619.
- McLean, R. D., and Pontiff, J. (2016). Does Academic Research Destroy
  Stock Return Predictability? *Journal of Finance* 71(1), 5–32.
- Moreira, A., and Muir, T. (2017). Volatility-Managed Portfolios. *Journal
  of Finance* 72(4), 1611–1644.
