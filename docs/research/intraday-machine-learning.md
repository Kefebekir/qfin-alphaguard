# Intraday machine learning for QFin-AlphaGuard: the evidence, our tests, and what to build

October 2026. This is the research behind decision 0018. Efe set its
direction: QFin-AlphaGuard is an intraday project, its ML must work within
the day, and he wants to see the system's choices day by day. It answers
three questions:
1. What can machine learning do within the day, for an account like ours
   (500 to 30,000 USD at Alpaca, a UK resident, a cash account at first)?
2. Which approach is best, judged by our own tests on our own data, after
   costs?
3. How can Efe see the choices day by day, and how can they be proven
   forward?

Every number from a paper was checked in the paper itself, unless marked as
coming from a summary. "Our test" numbers come from `research/intraday_pilot/`
and can be rerun. All our tests are walk-forward, with 2018 to October 2026
out of sample, unless stated.

## 1. The short answer

- **One intraday edge holds up for us: the market's own trend within the
  day, on QQQ** (decision 0011). Its rules earned 21.1% a year at a 2× cap
  with a Sharpe ratio of 1.61 out of sample in 2018–2026, and a Sharpe ratio
  of 1.26 after its paper's data ended (May 2024 – October 2026). It has a
  known cause (dealers hedging options in the direction of the day's move)
  and a published record.
- **Machine learning did not beat it, anywhere we tried.** On QQQ: choosing
  or sizing the day, making the half-hourly decisions, confirming the rule's
  decisions, and forecasting the next 15 minutes from 36 assets with the
  method of a 2025 *Management Science* paper. Every one did worse than the
  plain rule, or no better than holding QQQ.
- **Single stocks do have intraday signals, but not ones a small account can
  take.** They are a few basis points, smaller than the spread a small
  account pays near the open, or they live in the opening auction, which it
  cannot use. A model that ranks every S&P 500 member at 09:35 found a real
  signal (rank IC 0.030, t 9.5) and still lost 12.6 bps a day at the bid
  and ask.
- **So the intraday AI is a decision engine, and ML has to earn its place.**
  The rule makes the decisions; every decision is logged with its reason
  and shown day by day; ML models run beside it in shadow, logging what
  they would do, and trade only after beating the rule forward by a test
  written down in advance.
- **The day book exists now:** every QQQ decision since May 2024, 613
  sessions, with its chart and its reason (section 5).

This is a negative result for machine learning within the day, and it is
reported as it is.

## 2. What the literature shows

### 2.1 The market's own trend within the day

| Study | What it found |
| --- | --- |
| Gao, Han, Li and Zhou (2018), *Journal of Financial Economics* | SPY, 1993–2013: the first half hour's return predicts the last half hour's, with an R² of 1.6%. Stronger on volatile, high-volume, recession and news days, and in ten other liquid ETFs. (From the paper's abstract.) |
| Baltussen, Da, Lammers and Martens (2021), *Journal of Financial Economics* | The same pattern in index futures worldwide. The cause: dealers who are short options (gamma) must buy as the market rises and sell as it falls, in the direction of the day's move. |
| Zarattini, Aziz and Barbon (2024; revised 2025), SSRN 4824172 | The noise-area rule on SPY: long above a band built from the last 14 days' moves, short below it, with VWAP as a trailing stop, flat at the close. 2007 – April 2024: 19.6% a year net of costs, Sharpe ratio 1.33. Decision 0011 replicated it. |
| Paz (2026), SSRN 7290621 | An independent out-of-sample test of that SPY rule, May 2024 – March 2026, finds it degraded, and asks whether 0DTE options changed the market. (From a search summary; SSRN refused our download.) Our own test agrees: after May 2024 SPY's Sharpe ratio was 0.06 and QQQ's 1.16, both at caps up to 4× (decision 0011). |

### 2.2 Machine learning on the market's next minutes

| Study | What it found |
| --- | --- |
| Aleti, Bollerslev and Siggaard (2025), *Management Science* 71(9) | The 15-minute return of the US market, 1996–2020, predicted from the lagged 15-minute, hourly and daily returns of 272 factor and industry portfolios built from every stock. Ridge, lasso, elastic net, PCR, PLS, random forests, boosted trees, a neural network and their equal-weighted ensemble, refitted each year. Out of sample 2004–2020, the ensemble's R² is 0.21%; the AR models of the market alone are near zero. Trading SPY on the ensemble, changing side only when the forecast beats half the spread: Sharpe ratio 1.37 after costs, against 0.09 for holding SPY during the day. The gains come mostly in recessions and times of high uncertainty, and around 10:45 and 14:15–14:45, when macro data and FOMC decisions come out. Trading each ETF on its own forecast did markedly worse. |
| Huddleston, Liu and Stentoft (2023), *Journal of Financial Econometrics* 21(2) | 5-minute S&P 500 returns predicted from the lagged returns of its members. Ensembles give a Sharpe ratio of 0.98 after costs. Predictability fell after decimalisation, and is stronger mid-day, on volatile or illiquid days and in crises. (From the published abstract.) |
| Liu and Stentoft (2023), SSRN 4496917 | The same at the sector and single-stock level, 2004–2016: linear models predict best, nonlinear ones trade best after costs, and predictability is larger in less liquid stocks. (From summaries.) |
| Aït-Sahalia, Fan, Xue and Zhou (2022), NBER working paper 30366 | Predictability at the level of seconds is large and pervasive, and fades within minutes. That is the domain of firms with co-located servers. |

### 2.3 Single stocks within the day

| Study | What it found |
| --- | --- |
| Ghosh, Neufeld and Sahoo (2022), *Finance Research Letters* | S&P 500 members, 1993–2018. Each morning a random forest or LSTM, fed with 93 past returns (from opens and closes) and today's opening gap, ranks the stocks; buy the top 10 at the open and short the bottom 10, closed at the close. Before costs 0.64% a day (LSTM) and 0.54% (forest); after 5 bps a half-turn 0.44% and 0.34%. The returns were much smaller from 2010. |
| Zarattini, Barbon and Aziz (2024), SSRN 4729284 | The 5-minute opening range breakout on the 20 "stocks in play" (the most first-5-minute volume against normal), all US stocks above 5 USD, 2016–2023: 41.6% a year, Sharpe ratio 2.81. A stop loss 10% of the 14-day average true range from the entry. Costs are a commission of 0.0035 USD a share only, with no spread or slippage. On all stocks, without the "in play" filter: 3.2% a year, Sharpe 0.48. |
| Baltussen, Da and Soebhag (2025), working paper | End-of-day reversal: stocks that rose most from yesterday's close to 30 minutes before today's close fall in the last half hour, and losers rise. 1993–2019, long-short 3.78–6.86 bps a day, t-statistics above 10; driven by retail buying of the day's losers. |
| Heston, Korajczyk and Sadka (2010), *Journal of Finance* | A stock's return in a half hour predicts its return in the same half hour on following days, for weeks; returns in the next few half hours reverse (bid-ask bounce). The authors suggest using it to time trades, which saves about the effective spread. |
| Chinco, Clark-Joseph and Ye (2019), *Journal of Finance* | Lasso on the whole cross-section's lagged returns predicts one-minute-ahead returns; the signals are sparse and short-lived. |

### 2.4 Order books and deep learning

Kolm, Turiel and Westray (2023, *Mathematical Finance*) forecast returns from
order flow imbalance with LSTMs; the horizon is about two price changes.
Briola, Bartolucci and Aste (arXiv 2403.09267) find that high forecast
accuracy in order books often does not become a tradeable signal. Both need
full order-book data and microsecond execution, which we do not have.

### 2.5 Machine learning on top of a rule

- **Meta-labelling** (López de Prado, 2018): a primary rule picks the side,
  a secondary model decides whether to act and how much.
- **Volatility for sizing:** HAR models are hard to beat with ML for daily
  realised variance (decision 0017's research).

### 2.6 What costs and access mean for us

- **Alpaca's costs:** no commission; regulatory fees on sales; the spread.
  A marketable order of about 2,000 USD costs about 1.3 bps against the
  decision price mid-day (decision 0012), about twice that at 09:35
  (decision 0013).
- **Auction orders:** Alpaca's market-on-open and market-on-close orders
  (`opg`, `cls`) take whole shares only, and its order table marks them
  "contact the sales team". Fractional orders are day orders only. A small
  account cannot count on trading at the official open.
- **Spreads at the open, measured:** Alpaca's NBBO quotes for the S&P 500
  members that gap most show a half spread of 21 bps at 09:31, 12 bps at
  09:35 and 1.7 bps at 15:59 (2016–2026); among the 100 most traded, 9, 6
  and 1.3 bps. Trading near the open costs several times what it costs
  later in the day.
- **The pattern day trader rule is gone.** FINRA dropped it from Rule 4210
  on 4 June 2026; brokers have until 20 October 2027 to move to the new
  intraday margin checks. A margin account still needs 2,000 USD. A 500 USD
  account is a cash account: long only, no leverage. Alpaca covers unsettled
  sale proceeds, so a cash account can trade every day.
- **QQQ from the UK:** PRIIPs keeps US ETFs from UK retail investors; the
  route is a Nasdaq-100 CFD or spread bet (decision 0011).

## 3. Our tests

Data: the stored 1-minute bars of each year's 50 trading-universe stocks,
of QQQ and of 15 more ETFs; the daily bars of every S&P 500 member; and,
for this research, the first 16 minutes of every member's session and the
consolidated tape's quotes at 09:31, 09:35 and 15:59 for the stocks the
morning tests pick, all from Alpaca's SIP history. Costs are stated per test.

### 3.1 The market: QQQ within the day

| Test | Result |
| --- | --- |
| **The noise-area rule** (decision 0011), re-implemented, cap 2× | 2016–2026: 17.9% a year, Sharpe ratio 1.43 (decision 0011's research code, not in the repository: 18.3%, 1.47). 2018–2026: 21.1% a year, Sharpe 1.61, worst drawdown −12.8%. After the paper's data end (May 2024 – October 2026): +44.5% in all, Sharpe 1.26, worst drawdown −12.8%; it traded on 56% of days and won on 26%. |
| **ML at 10:00 skips or resizes the day.** Boosted trees and ridge on 15 features known at 10:00 (the gap, the first half hour's move, range and volume, the band, recent volatility, the strategy's own last days) | Skipping the days the model ranks in its bottom third: −2.6% and −4.9% a year against the rule (t −1.12 and −2.13). Half or 1.5× the size instead: +1.3% (t 0.92) and −0.5% (t −0.33). No significant gain in any of the four; the best one's t moved from 0.35 to 0.92 when three days of input changed (the missing-minute fix of step 2), which is what noise looks like. |
| **HAR volatility for the size** | HAR forecasts the day's volatility better than the rule's 14-day measure (correlation with the realised value 0.72 against 0.61), but the strategy earns 20.6% a year and Sharpe 1.57 with it, against 21.1% and 1.61. No gain. |
| **ML makes the half-hourly decisions.** Boosted trees and ridge on 13 features at each HH:00 and HH:30: the price against the band's edges and VWAP, the move since the open and in the last half hour, the gap, the time, the volume and volatility so far, the rule's position | Alone: −0.2% and 7.2% a year (Sharpe 0.07 and 0.51), right on 50.4% and 51.5% of the half hours it held a position; the rule is right on 53.5%. As a meta-label that keeps the rule's position only when the model agrees: Sharpe 1.27 and 1.48 against 1.61 (−7.6% and −2.7% a year, t −3.11 and −1.92). Worse. |
| **The factor-zoo method** (Aleti, Bollerslev and Siggaard) on what we store: the last 15 minutes, the last hour, the day so far and the opening gap of 36 assets (QQQ; SPY, IWM, DIA, TLT, GLD, HYG and the nine sector SPDRs; the 20 stocks in every year's trading universe), predicting QQQ's next 15 minutes. Ridge, lasso, elastic net, PCR, PLS, boosted trees, a neural network and their ensemble, tuned and refitted each year | No forecasting power. The out-of-sample R² is below zero for every model but the trees (+0.005%), and −0.007% for the ensemble; the sign is right on 50.4–50.9% of the 15-minute slots. The ensemble's R² was positive in 2020, 2021 and 2024 (+0.07%, +0.12%, +0.04%) and negative in the six other years; the paper, too, found its gains in times of crisis. Traded the paper's way, changing side only when the forecast beats one side's cost: 2–10% a year, Sharpe 0.13–0.65 across the models, the ensemble 0.45 and 0.55, against 0.37 for simply holding QQQ over the same hours before costs. Far below the rule. |

### 3.2 Single stocks within the day

| Test | Result |
| --- | --- |
| **End-of-day reversal** (Baltussen, Da and Soebhag), our 50 stocks, 2016–2026 | The pattern is there: the day's biggest losers beat its biggest winners in the last half hour by 2.77 bps a day (t 5.8). Buying the 5 biggest losers 30 minutes before the close: +1.45 bps a day before costs, −0.55 after a 2 bps round trip. It has faded: −1.78, +0.53 and −0.49 bps a day of excess return in 2024, 2025 and 2026. |
| **10:00 to the close.** Ridge and boosted trees rank our 50 stocks on 13 features (the gap, the first half hour, how much a stock is in play, recent returns and volatility, and the same against the universe) | Rank IC 0.017 (t 3.3) and 0.012 (t 3.1). Buying the top 5 at 10:00 and selling at the close: −0.2% and +2.3% a year after a 2 bps round trip (Sharpe −0.01 and 0.12), −2.7% and −0.2% after 3 bps; the top 5 beat the 50 by 1.0 and 2.0 bps a day before costs (t 0.7 and 1.5). |
| **Opening range breakout on stocks in play** (Zarattini, Barbon and Aziz), our 50 stocks, the 5 most in play each day | With the paper's filters (price above 5 USD, a million shares a day, an ATR above 0.50 USD). The stop sits only 0.27% from the entry on average, so the result depends on whether a 1-minute bar's low came before or after the entry, which the bar cannot tell. Before costs, long and short, it ranges from −38% to +13% a year between the two readings; with 2 bps of slippage on the entry and on the exit, from −55% to −18% long and short, and from −27% to −1% long only. Fails. |
| **Ghosh, Neufeld and Sahoo on every S&P 500 member,** open to close, about 500 stocks a day | Random forest and boosted trees on the paper's 93 features: rank IC 0.030 and 0.025 (t 10). But today's opening gap alone has 0.034 (t 8): the models learned the gap reversal and added nothing to it. Ten long and ten short, at the official open and close: Sharpe 1.52 and 1.28 before costs, 0.68 and 0.41 after 2 bps a half-turn, below zero after 5 bps; the gap alone 1.64, 1.00 and 0.03. |
| **Is the gap reversal in a price one can trade at?** The same stocks bought at the first trade from 09:31, 09:35 or 09:45 instead of the official open, 2016–2026, 1.26 million member-days | No. Ten long and ten short earn 17.7 bps a day from the official open (t 4.9), but only 4.3 bps from 09:31 (t 1.2), 4.0 from 09:35 and 3.9 from 09:45. The long side alone falls from +6.8 bps over all members to −0.6. The reversal happens in the opening auction and its first minute; a small account at Alpaca cannot use auction orders, and the spread at 09:31–09:35 is larger than what is left: the stocks it buys have a half spread of 21 bps at 09:31 and 12 bps at 09:35 (NBBO quotes), and bought at the ask at 09:35 and sold at the bid at 15:59, they lose 14.6 bps a day (t −5.0); among the 100 most traded, half spreads of 9 and 6 bps and a loss of 6.8 bps a day. Among the 100 most traded members: 8.2 bps from the open (t 2.6), 3.7 from 09:31, 1.6 from 09:35. |
| **At 09:35, which members will rise most by the close?** Ridge and boosted trees on 15 features known at 09:35: the gap; the first 5 minutes' move, range and volume against the same 5 minutes of the last 14 sessions; recent returns and volatility; the same against all members; size and price. Every S&P 500 member, about 500 a day | The signal is real before costs. The trees' rank IC is 0.030 (t 9.5), against 0.013 for the gap alone from 09:35. Their daily top 10 earn 5.5 bps a day from the first trade after 09:35 to the official close, 5.1 more than all members (t 2.6); among the 100 most traded, 3.4 and 2.7 (t 1.7). At the prices a small account pays, it loses. The stocks it picks have a half spread of 14.3 bps at 09:35 (6.5 among the 100 most traded): part of what it learned is the bounce between the bid and the ask. From the 09:35 mid to the 15:59 mid they earn 3.8 bps a day (t 1.3); bought at the ask and sold at the bid, they lose 12.6 bps a day, 32% a year (5.1 bps among the 100 most traded); sold at the closing auction instead, 10.6 and 3.6. It lost in 8 of the 9 years in both groups. |

### 3.3 What the tests say together

- **One robust intraday edge in our data: the market's own trend, on QQQ.**
  It is a simple rule with a published mechanism (dealers' gamma hedging),
  it held up after its publication, and no model we tried improved it.
- **Machine learning added nothing that survives costs,** at any of the
  places it could act: choosing the day, sizing it, making the half-hourly
  decisions, ranking stocks in the morning or at 10:00, or forecasting the
  next 15 minutes from other assets.
- **The single-stock signals are real but live where we cannot trade:** in
  the opening auction and its first minute, or in the last half hour at
  sizes of 1–2 bps, below the spread. The breakout on stocks in play may
  live in smaller stocks with news, which our data do not cover.
- **Our own numbers fit the literature's caveats:** predictability fell
  after decimalisation, after publication, and in large, liquid stocks; the
  papers that find tradeable intraday ML (Aleti, Bollerslev and Siggaard;
  Huddleston, Liu and Stentoft) end in 2020 and 2016 and earn most in crises.

### 3.4 The rules in the engine (Phase 2, step 2)

`research/intraday_pilot/engine_check.py` runs the strategy of
`src/qfin_alphaguard/momentum.py` through the backtest engine on the stored
QQQ minutes, 2016–2026, from 30,000 USD, with a stand-in that lets every
order through (Guard's limits for it are step 4):

- **The same decisions.** It trades on exactly the same 1,580 days as
  `qqq_momentum.py`. At no cost it earns 19.5% a year with a Sharpe ratio of
  1.57, against 19.6% and 1.56, and the daily results correlate at 0.996.
  What differs is by design: the engine leaves on the last minute's open,
  not at the close (on the few days QQQ moved 36–47 bps in the last minute,
  that is up to 95 bps at 2×), and it turns from long to short a minute
  later, because Alpaca takes no order that crosses zero.
- **A bug found in the research.** The scripts left a missing minute empty
  instead of carrying the last price, because polars' `fill_null` does not
  fill NaN. QQQ lacks minutes on four days, the trading halts of March 2020;
  for 14 sessions after each, no decision was taken at those minutes. Fixed;
  three days' results changed, and the numbers above are after the fix.
- **QQQ's spread, measured.** Alpaca's NBBO quotes on 101 days of 2016–2026
  show a spread of a cent: a half spread of 0.36 bps at 10:00 and 0.23 bps at
  15:59, about 0.3 on average. With it and Alpaca's fees, the engine earns
  16.2% a year with a Sharpe ratio of 1.33; the research's cost model had
  assumed less spread (17.9% and 1.43). The simulated broker's default half
  spread of 1 bp, measured for the trading universe's stocks (decision
  0012), is three times QQQ's and would give 11.1% and 0.95: the broker
  needs a spread for each instrument.

## 4. What to build

1. **The QQQ strategy in our engine,** as its own `Strategy`: the noise
   band, VWAP, decisions at 10:00 and every half hour, a size of
   min(2, 2% / 14-day volatility), flat at the close. Efe writes the rules;
   Claude prepares the skeleton, the spec and the tests. It must reproduce
   `qqq_momentum.py` day by day.
2. **A decision log** written by the strategy, the same in a backtest, in
   paper trading and live (section 5.2), and the day book reading it.
3. **Guard's limits for an intraday strategy,** set by Efe in `guard.yaml`:
   round trips and reversals allowed for it, the 2× cap, the daily loss
   limit (decision 0011).
4. **ML challengers in shadow** (section 5.3), each frozen and with its test
   written down before its record starts, first on backtest replays:
   - the factor-zoo ensemble of section 3.1, the literature's best-documented
     intraday ML, whose gains in the paper came mostly in crises;
   - a model that sizes the QQQ day (section 3.1's skip-or-resize models);
   - the 09:35 stock ranker of section 3.2.

   None is expected to win. Recording them costs little and shows, day by
   day, what the AI would have done.
5. **Then Phase 3:** paper trading on Alpaca for two to three months, with
   the challengers in shadow, the day book every evening and a replay of
   each day; and in Phase 3b a broker and instrument for real money from
   the UK (decision 0011).

Not built, on this evidence: order-book models (no data, and no connection
fast enough), stocks-in-play breakouts, the end-of-day reversal, and
open-to-close stock ML. Decision 0017's monthly ranking stays as research
for the nightly plan.

## 5. Seeing the choices day by day, and proving them forward

### 5.1 The day book

`research/intraday_pilot/daily_choices.py` writes a page for the browser
(`data/reports/qqq_day_book.html`):

- **the days,** month by month, each with its result after costs, and the
  totals: return, Sharpe ratio, worst drawdown, the share of days won and
  traded;
- **a day's chart:** QQQ's price, the noise band, VWAP, and a dot at each
  fill (green: now long, red: now short, grey: now flat);
- **a day's decisions,** at 10:00, 10:30 and every half hour to 15:30, and
  the close a minute before the end: the price, the band, VWAP, the
  position chosen, the fill, and why ("above the band: buy", "long stopped:
  price fell below the band's top or VWAP", "the trend holds; its stop is
  at 735.84").

The page holds licensed prices, so it stays in the git-ignored `data/` and
is for Efe's own use.

### 5.2 The decision log (Phase 2, step 3)

Since step 3 the page no longer replays research code: it runs the engine
and reads what the strategy logged.

- **A DecisionNote for every decision,** a kept position included: its time,
  what the strategy saw (price, band, VWAP, shares, volatility, the
  account), the choice, and the reason in words. The engine writes it into
  the log before the orders it explains; Guard never sees it.
- **Stored as tables** (`decision_log.py`): the notes and the fills, as
  Parquet. The live engine is to write the same tables, so a paper day and a
  backtest day are read the same way.
- **The same decisions as the research.** Of the research's 32,190
  half-hourly decisions in 2016–2026, 32,189 are the same in the log. The
  other fell in the market-wide halt of 18 March 2020: the engine decided on
  the first minute after trading resumed.
- **Each evening of paper trading** (Phase 3), the day's page and a replay of
  the same day from the stored minutes are to be compared; Gate M3 asks that
  a replay reproduces at least 95% of the paper decisions.

### 5.3 Proving it forward

An intraday decision cannot be committed before the open; the choice exists
only once the morning's prices do. What can be fixed in advance is how the
choices are made:

1. **Freeze the rules and any model:** the commit, the parameters, and for a
   model the training dates and the hash of its weights.
2. **Commit each evening's decision log** to the repository, with its hash,
   next to the paper account's own fill records from Alpaca. Anyone with the
   minutes can replay the frozen rules and get the same decisions.
3. **ML challengers in shadow:** a challenger logs the forecast and the
   position it would take at every decision, in the same log, without
   trading. Its test is written down before its record starts: the paired
   daily difference against the rule, with a t-statistic above 2 after
   counting every challenger tried.
4. **Judge by what holds forward:** the daily result against the backtest's
   5–95% range for the same number of days.

How long proof takes, with t ≈ Sharpe × √years:

| Daily Sharpe ratio, a year | Years to t = 2 |
| --- | --- |
| 1.24 (QQQ, out of sample, decision 0011) | 2.6 |
| 1.0 | 4.0 |
| 0.5 (a challenger's edge over the rule) | 16 |

So the forward record proves first that the system trades as its backtest
says (two to three months of paper trading). Proving the edge itself takes
years, and a small improvement from a challenger cannot be proven in any
reasonable time; only a large one can.

## References

- Aït-Sahalia, Y., Fan, J., Xue, L., and Zhou, Y. (2022). How and When Are
  High-Frequency Stock Returns Predictable? NBER Working Paper 30366.
- Aleti, S., Bollerslev, T., and Siggaard, M. (2025). Intraday Market Return
  Predictability Culled from the Factor Zoo. *Management Science* 71(9),
  7731–7751.
- Baltussen, G., Da, Z., Lammers, S., and Martens, M. (2021). Hedging Demand
  and Market Intraday Momentum. *Journal of Financial Economics* 142(1),
  377–403.
- Baltussen, G., Da, Z., and Soebhag, A. (2025). End-of-Day Reversal.
  Working paper.
- Briola, A., Bartolucci, S., and Aste, T. Deep Limit Order Book
  Forecasting: A Microstructural Guide. arXiv 2403.09267.
- Chinco, A., Clark-Joseph, A. D., and Ye, M. (2019). Sparse Signals in the
  Cross-Section of Returns. *Journal of Finance* 74(1), 449–492.
- Gao, L., Han, Y., Li, S. Z., and Zhou, G. (2018). Market Intraday Momentum.
  *Journal of Financial Economics* 129(2), 394–414.
- Ghosh, P., Neufeld, A., and Sahoo, J. K. (2022). Forecasting Directional
  Movements of Stock Prices for Intraday Trading Using LSTM and Random
  Forests. *Finance Research Letters* 46, 102280.
- Heston, S. L., Korajczyk, R. A., and Sadka, R. (2010). Intraday Patterns in
  the Cross-Section of Stock Returns. *Journal of Finance* 65(4), 1369–1407.
- Huddleston, D., Liu, F., and Stentoft, L. (2023). Intraday Market
  Predictability: A Machine Learning Approach. *Journal of Financial
  Econometrics* 21(2), 485–527.
- Kolm, P. N., Turiel, J., and Westray, N. (2023). Deep Order Flow Imbalance:
  Extracting Alpha at Multiple Horizons from the Limit Order Book.
  *Mathematical Finance* 33(4), 1044–1081.
- Liu, F., and Stentoft, L. (2023). Intraday Stock Predictability
  Everywhere. SSRN 4496917.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley.
- Paz, S. (2026). Out-of-Sample Evaluation of an Intraday Momentum Strategy
  for the S&P 500 ETF (SPY). SSRN 7290621.
- Zarattini, C., Aziz, A., and Barbon, A. (2024). Beat the Market: An
  Effective Intraday Momentum Strategy for S&P500 ETF (SPY). SSRN 4824172.
- Zarattini, C., Barbon, A., and Aziz, A. (2024). A Profitable Day Trading
  Strategy for the U.S. Equity Market. SSRN 4729284.
