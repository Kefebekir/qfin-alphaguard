# Source pool for Phase 2 step 5: where intraday or short-horizon edges come from

Each line: the source, what it finds, and what it means for us (QQQ, minute bars, a small
Alpaca account). "Test" marks what our own data can check. Grouped by theme; added as the
search goes on (started 2026-10-15).

## A. How professionals make money (expectation setting)
1. Zuckerman, *The Man Who Solved the Market* (2019): Mercer, "right 50.75% of the time" over thousands of bets. https://www.goodreads.com/notes/46025922-the-man-who-solved-the-market/13424516-zubair-habib/bd279716-8be3-44b0-849d-d5527dbb7cd6 — edge per bet tiny; breadth everything.
2. Virtu S-1 (2014): 1 losing day in 1,238 (2009-2013), from spread capture and risk control, not prediction. https://qz.com/186209/the-high-frequency-trading-firm-thats-about-to-go-public-only-had-one-day-of-trading-losses-in-four-years
3. Grinold, fundamental law IR = IC x sqrt(breadth); critiques: estimation error, constraints, correlated bets (Michaud et al. 2020). https://newfrontieradvisors.com/media/1744/fundamental-law-september-2019.pdf — our breadth is tiny (one index, ~12 decisions a day, highly dependent).

## B. Intraday momentum and its mechanism
4. Gao, Han, Li, Zhou (JFE 2018), market intraday momentum: first half hour predicts last; stronger on volatile, high-volume, news days. https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351 — Test: done (payrolls days +, CPI -).
5. Baltussen, Da, Lammers, Martens (JFE 2021), hedging demand: momentum on negative dealer-gamma days, none on positive; 60+ futures. https://www3.nd.edu/~zda/intramom.pdf — the mechanism; real gamma test pre-registered.
6. Barbon, Beckmeyer, Buraschi, Moerke, LETF and option rebalancing flows: end-of-day momentum; LETF effect shrinking over time. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3925725
7. Dim, Eraker, Vilkov, 0DTEs: positive dealer gamma -> reversal, negative -> momentum; short-lived. https://papers.ssrn.com/sol3/Delivery.cfm/4692190.pdf?abstractid=4692190
8. Zarattini, Aziz, Barbon, Beat the Market (SPY intraday momentum, noise area) — our rule's source (decision 0011).
9. Mesfin (arXiv 2605.04004, 2026): 14 OHLCV intraday momentum families on Nasdaq micro futures 2021-2025; none survive costs. https://arxiv.org/abs/2605.04004 — agrees with our ML failures.
10. Shen, Urquhart, Wang (Financial Review 2022): Bitcoin intraday momentum, driven by liquidity provision. https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/
11. Heston, Korajczyk, Sadka (JF 2010): half-hour continuation at daily multiples for 40 days, strongest first/last half hour. https://arxiv.org/abs/1005.3535 — Test candidate: QQQ same-half-hour seasonality.
12. Rosa (J Futures Markets 2022): overnight return predicts last half hour in sample, not out of sample. https://onlinelibrary.wiley.com/doi/10.1002/fut.22375

## C. Machine learning in finance: where it works and where it does not
13. Gu, Kelly, Xiu (RFS 2020): trees and neural nets best; NN long-short value-weighted Sharpe 1.35; S&P timing 0.77 vs 0.51. https://papers.ssrn.com/abstract=3159577
14. Avramov, Cheng, Metzker: ML gains concentrated in microcaps, unrated and distressed firms. https://papers.ssrn.com/abstract=3306110
15. Kelly, Malamud, Zhou (JF 2024), virtue of complexity. https://www.sfi.ch/en/publications/the-virtue-of-complexity-in-return-prediction
16. Nagel (NBER w34104, 2025): that complexity is volatility-timed momentum that happened to work. https://www.nber.org/papers/w34104
17. Jiang, Kelly, Xiu (JF 2023), CNN on price-chart images; Sharpe ~1 in the 500 largest stocks, far higher in small ones. https://papers.ssrn.com/abstract=3756587
18. Ke, Kelly, Xiu, predicting returns with text (SESTM, Dow Jones newswires): slow assimilation, tradeable net of costs, stronger in small volatile firms. https://www.nber.org/papers/w26186
19. Lopez-Lira, Tang: ChatGPT headline scores predict next-day returns; edge shrinks as LLM adoption grows. https://arxiv.org/abs/2304.07619v5
20. Chen, Pelger, Zhu (Mgmt Sci 2024): deep learning SDF with no-arbitrage. https://arxiv.org/pdf/1904.00745
21. Gu, Kelly, Xiu (J Econometrics 2021): autoencoder asset pricing. https://papers.ssrn.com/abstract=3335536
22. Aleti, Bollerslev, Siggaard (Mgmt Sci 2025): intraday market return predictability from 200+ factors' lagged returns. https://public.econ.duke.edu/~boller/Papers/MS_2025.pdf — Test: done (no power on our 36 assets).
23. Ait-Sahalia, Fan, Xue, Zhu, how and when high-frequency returns are predictable: large but very short horizon. https://www.nber.org/system/files/working_papers/w30366/w30366.pdf
24. Ferreira, Medeiros (arXiv 2112.15108): minute SPY, LSTM vs RF; VIX the strongest predictor. https://arxiv.org/abs/2112.15108
25. Lim, Zohren, Roberts (2019) Deep Momentum Networks. https://arxiv.org/abs/1904.04912 — Test: done (policy_nets.py).
26. Wood et al. Momentum Transformer (2022); X-Trend (2023); DeePM (2026). https://arxiv.org/abs/2112.08534v3 https://arxiv.org/pdf/2310.10500 https://www.arxiv.org/abs/2601.05975
27. Wood, Roberts, Zohren, Slow Momentum with Fast Reversion (changepoint detection). https://arxiv.org/pdf/2105.13727
28. Moody, Saffell (1998/2001) direct reinforcement, differential Sharpe ratio. https://papers.nips.cc/paper/1551-reinforcement-learning-for-trading
29. Zhang, Zohren, Roberts (2020) deep RL for trading futures. https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/06/Deep-Reinforcement-Learning-for-Trading.pdf
30. Rahimikia et al.: time-series foundation models in finance lose to CatBoost; negative out-of-sample R2. https://arxiv.org/html/2511.18578v1
31. Buehler et al., data-driven market simulator (signatures); no shown out-of-sample trading gain. https://arxiv.org/pdf/2006.14498
32. Guijarro-Ordonez, Pelger, Zanotti, deep learning statistical arbitrage. https://arxiv.org/abs/2106.04028

## D. LLM trading agents (and why their alpha is doubtful)
33. Profit Mirage (arXiv 2510.07920): Sharpe decays 51-62% out of sample. https://arxiv.org/html/2510.07920
34. FINSABER (arXiv 2505.07078): LLM strategies trail passive benchmarks over long, broad tests. https://arxiv.org/html/2505.07078v5
35. The Alpha Illusion (arXiv 2605.16895). https://arxiv.org/html/2605.16895
36. Look-ahead in model weights: LLMs recall prices (arXiv 2605.24564). https://arxiv.org/pdf/2605.24564

## E. Calendar and announcement effects
37. Lucca, Moench (JF 2015) pre-FOMC drift; Kurov, Wolfe, Gilbert (FRL 2021): gone after 2015. https://www.skidmore.edu/economics/documents/KurovWolfeGilbert-TheDisappearingPre-FOMC-Announce-Drift-200914.pdf — Test: done; back in 2020-2026 (+41 bps, t 2.88).
38. Boyarchenko, Larsen, Whelan, the overnight drift (2-3 am ET); faded since 2021 (NY Fed 2026). https://libertystreeteconomics.newyorkfed.org/?p=43261
39. Lou, Polk, Skouras (JFE 2019), tug of war overnight vs intraday. https://personal.lse.ac.uk/polk/research/TugOfWar.pdf
40. Etula, Rinne, Suominen, Vaittinen (RFS 2020), dash for cash, turn of the month. https://academic.oup.com/rfs/article/33/1/75/5494694 — Test: done (not significant).

## F. Replication and decay
41. McLean, Pontiff (JF 2016): 26% lower out of sample, 58% lower after publication. https://ivey.uwo.ca/media/3775549/pontiff.pdf
42. Jensen, Kelly, Pedersen (JF 2023): most factors replicate, 13 themes. https://www.nber.org/papers/w28432

## G. Options, volatility, liquidity provision
43. Nagel (RFS 2012), evaporating liquidity: reversal returns = liquidity provision, predictable by VIX. https://nber.org/papers/w17653
44. Beckmeyer, Branger, Gayda: retail 0DTE buyers lose; short side earns variance premium. https://medium.com/@y125789/retail-traders-love-0dte-options-paper-summary-6da4a7ff7d72
45. Wysocki (arXiv 2508.16598) put-writing sizing (Kelly, VIX). https://arxiv.org/abs/2508.16598
46. Moskowitz, Ooi, Pedersen (JFE 2012) time-series momentum; Hurst, Ooi, Pedersen, a century of trend following. — Test: done on 16 ETFs (Sharpe 0.33-0.55; long/flat 0.74-0.83).

## H. Data sources found
47. Databento OPRA (statistics: open interest, 2013+). https://databento.com/datasets/OPRA.PILLAR
48. ThetaData (options from 2016 on Standard). https://thetadata.net/docs/Articles/Getting-Started/Subscriptions.html
49. WRDS OptionMetrics (1996+, strike-level OI). https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics
50. Cboe index histories (VIX, VIX9D, VIX1D, VIX3M, VVIX, VXN): downloaded to data/raw/cboe.

## I. Microstructure and order flow
51. Cont, Kukanov, Stoikov (J Fin Econometrics 2014): price changes linear in order flow imbalance, slope 1/depth. https://arxiv.org/abs/1011.6402 — needs quotes; QQQ's are huge.
52. Sirignano, Cont (Quant Finance 2019): universal price formation, deep net on order books, pooled across stocks beats per-stock. https://arxiv.org/pdf/1803.06917 — seconds-ahead, needs order-book data and speed.
53. Bouchaud, Bonart, Donier, Gould, *Trades, Quotes and Prices* (2018); Lillo-Farmer: order signs long memory (metaorder splitting); square-root impact. — explains why flows persist intraday.
54. Chordia, Roll, Subrahmanyam (JFE 2002): market returns respond to contemporaneous and lagged order imbalance; reversal after extreme selling days. https://www.cis.upenn.edu/~mkearns/finread/stock-misbalance.pdf
55. Chordia, Subrahmanyam (JFE 2004): lagged stock imbalance predicts returns; costs eat it. 
56. Bacry, Mastromatteo, Muzy (2015) Hawkes processes in finance: modelling, execution; no shown trading gain. https://alphaxiv.org/abs/1502.04592
57. Bogousslavsky, Muravyev, who trades at the close: closing auctions 3.1% of volume in 2010, 7.5% in 2018; close-price deviations reverse overnight. https://thepluto.substack.com/api/v1/file/06bdf231-3aea-4e3a-8ca1-d5ef6ead54f9.pdf — Test: last half hour reverses overnight in 2016-2026 (t -9) but almost all from 2020; absent 2014-2015; validation failed.
58. Zhao (arXiv 2608.03703, 2026), preying on leveraged ETFs (Korea): speculators front-run closing LETF rebalances. https://arxiv.org/abs/2608.03703

## J. Retail and fund flows
59. Boehmer, Jones, Zhang, Zhang (JF 2021): retail order imbalance (subpenny) predicts ~10 bps a week. https://papers.ssrn.com/author=3572085 — cross-sectional, weekly.
60. Barber, Huang, Odean, Schwarz (JF 2022): Robinhood herding predicts -4.7% over 20 days. https://iimb.ac.in/sites/default/files/inline-files/attention-induced-trading.pdf
61. Gabaix, Koijen, inelastic markets: $1 of flow moves the market by ~$5. https://www.nber.org/system/files/working_papers/w28967/w28967.pdf — flows matter for prices.
62. Ben-David, Franzoni, Moussawi (JF 2018): ETF ownership raises stock volatility. https://www.nber.org/papers/20071
63. Ivanov, Lenkey (Fin Markets 2018): LETF late-day effect economically insignificant after capital flows. https://ideas.repec.org/a/eee/finmar/v41y2018icp36-56.html

## K. Options-based information
64. Pan, Poteshman (RFS 2006): open-buy put-call ratio predicts next-day stock returns (40 bps). https://www.nber.org/papers/w10925 — needs customer open/close data (CBOE).
65. Xing, Zhang, Zhao (JFQA 2010): steep smirk underperforms 10.9% a year; halved in recent data. 
66. Ni, Pearson, Poteshman, White (RFS 2021): option market makers' hedge rebalancing moves volatility. https://ideas.repec.org/a/oup/rfinst/v34y2021i4p1952-1986..html
67. Ni, Pearson, Poteshman (JFE 2005): closes pinned at strikes on expiry days (16.5 bps). Golez, Jackwerth (2012): S&P futures pinned too. — agrees with the rule's weaker monthly expiry days.

## L. Calendar, seasonality, sentiment, and the unexpected
68. Heston, Sadka (JFE 2008); Keloharju, Linnainmaa, Nyberg (JF 2016): same-calendar-month seasonality, 13% a year cross-sectionally. 
69. Hirshleifer, Shumway (JF 2003): morning sunshine and returns, 26 countries; costs remove gains.
70. Edmans, Garcia, Norli (JF 2007): national soccer losses -> -49 bps next day; replication mixed.
71. Edmans, Fernandez-Perez, Garel, Indriawan (JFE 2022): music sentiment (Spotify positivity) predicts next-week reversal. https://lbsresearch.london.edu/2142
72. Preis, Moat, Stanley (2013) Google "debt" searches; Moat et al. Wikipedia: in-sample, data-snooped (non-financial words also "work"). https://arxiv.org/pdf/1403.1715
73. Savor, Wilson (JFQA 2013): macro announcement days earn >60% of the equity premium. — Test: overnight into CPI/payrolls mornings higher but not significant (t 0.36 QQQ).
74. Krohn, Mueller, Whelan (JF 2024): FX W-shaped returns around fixes, dealer hedging, every year of 20. https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf
75. Evans, the WM/Reuters 4pm fix. https://pages.stern.nyu.edu/~jh4/SternMicroMtg/Old/SternMicroMtg2015/Papers/4ExEvans.pdf
76. Lou, Yan, Zhang (RFS 2013): Treasury prices fall before auctions, rebound after (dealer capacity).

## M. Crypto
77. Liu, Tsyvinski (RFS 2021): crypto time-series momentum and attention. https://papers.ssrn.com/abstract=3226806
78. Perpetual funding-rate carry (vendor material; no academic net-return study found).

## N. Overfitting, regimes, volatility timing
79. Bailey, Borwein, Lopez de Prado, Zhu: probability of backtest overfitting. https://papers.ssrn.com/abstract=2326253
80. Bailey, Lopez de Prado: deflated Sharpe ratio. https://papers.ssrn.com/abstract=2460551
81. Harvey, Liu, Zhu (RFS 2016): new factors need t > 3. https://www.nber.org/papers/w20592
82. Ang, Timmermann (2012) regime changes; HMM timing out of sample: modest, cost-sensitive. https://www.nber.org/papers/w17182
83. Moreira, Muir; Cederburg, O'Doherty, Wang, Yan (JFE 2020): volatility management fails in real time (103 strategies). https://www.lehigh.edu/~xuy219/research/COWY.pdf — agrees with our HAR sizing result.

## O. Evaluating a challenger forward: sequential, anytime-valid tests
84. Shafer, Vovk, testing by betting; Ramdas, hypothesis testing with e-values, martingales and betting (slides). https://stat.cmu.edu/~aramdas/talks/JHU24.pdf — a challenger "bets" against the rule each day; promote when its wealth passes 1/alpha; valid however often one looks (Ville).
85. Waudby-Smith, Ramdas (JRSS-B 2024): estimating bounded means by betting; confidence sequences. https://arxiv.org/abs/2010.09686v4 — the method for the shadow test.
86. Merging sequential e-values via martingales. https://arxiv.org/pdf/2007.06382 — combining many challengers honestly.
87. Harvey, Liu, evaluating trading strategies (haircut Sharpe ratios). https://www.stat.berkeley.edu/~aldous/157/Papers/harvey.pdf

## P. Networks, paths, cross-asset structure
88. Cohen, Frazzini (JF 2008) economic links: customer momentum; smaller after publication. https://arxiv.org/pdf/2301.11394
89. DeltaLag (arXiv 2511.00390): learned lead-lag pairs by cross-attention. https://arxiv.org/pdf/2511.00390
90. Futter, Horvath, Wiese (2023) signature trading: path-dependent mean-variance with signals. https://arxiv.org/pdf/2308.15135
91. Harris (JFE 1986): intraday return patterns (first 45 minutes, last 15 minutes); U-shaped volatility.

## Q. Dealer gamma: measurement caveats
92. "Your GEX chart is a model, not a position report" (Chilingarian 2026 summary): reversing the long-call/short-put convention flips the sign of GEX. https://dn12448583.substack.com/p/your-gex-chart-is-a-model-not-a-position — our pre-registered test uses the naive convention; a failure would not refute the mechanism.
93. FOMO: FLEX options and market outcomes (arXiv 2610.13044). https://arxiv.org/html/2610.13044v1
94. Barbon, Buraschi, gamma fragility (dealer gamma imbalance and intraday momentum/reversal). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3725454

## R. The 0DTE era (a possible reason for the 2022 regime change)
95. Adams, Dim, Eraker, Fontaine, Ornthanalai, Vilkov (SSRN 5641974), do S&P 500 options increase volatility? Evidence from 0DTEs: market makers' intraday hedging needs predict stronger reversals and lower momentum; the effect comes from longer-dated positions that become 0DTE. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5641974
96. Exchange calendar: SPX Tuesday/Thursday expiries from 18 April and 11 May 2022; QQQ and SPY Tuesday/Thursday expiries from 14-15 November 2022 (MIAX alert). https://miaxglobal.com/alert/2022/11/14/miax-exchange-group-options-markets-listing-tuesday-and-thursday-weekly-1 — our free gamma proxies reversed from 2023: consistent, but post hoc.
97. dev.to (firm post, not reviewed): last-half-hour momentum flat on 1,085 SPX sessions of the 0DTE era. https://dev.to/firmtape/intraday-momentum-is-dead-in-the-0dte-era-we-measured-it-on-1085-spx-sessions-43g0 — yet our noise-area rule kept a Sharpe of 1.55 in 2023-2026.

## S. Breadth for intraday momentum (our tests)
- The rule on 97 large stocks: no edge before costs (median Sharpe ~0); intraday momentum is an index phenomenon.
- On the 15 other ETFs: before costs only SPY (1.01) and XLK (1.09), both 0.67-0.80 correlated with QQQ; bonds, gold and sectors ~0. No cross-asset breadth from this rule.

## Status of every testable idea (2026-10-15), each test written down before it ran

| Idea (sources) | Our test | Result |
|---|---|---|
| Learned trading policies: DMN, Momentum Transformer, GRU, gate, day sizer, kNN, linear Sharpe (25-32) | policy_nets.py, 26 variants, 2019-22 select, 2023-26 holdout | failed; the one that passed selection lost in the holdout |
| Price-based gamma proxies: variance ratio, realised/implied (5-7, 94) | gamma_test.py H1/H2 on 2023-26 | failed; the relation reversed after 2022 |
| Online expert allocation (Hedge, Fixed-Share) (method) | hedge_test.py | failed; too slow to follow regime changes |
| Event days: FOMC, CPI, payrolls, expiries (4, 37, 67, 73) | flow_explore.py (exploration) | consistent with theory, weak (t 1.5-2.2) |
| Market breadth at 09:45 (none found in literature) | exploration | not monotone: no |
| Timing luck, a 15-minute shifted grid (method) | exploration | the shifted grid is worse (Sharpe 1.18 -> 0.93) |
| Overnight vs intraday; overnight leg beside the rule (38, 39) | calendar_effects.py | overnight ~60% of QQQ's return, corr +0.02 with the rule; rule + 0.5x overnight Sharpe 1.31 -> 1.51 |
| Pre-FOMC drift (37) | calendar_effects.py | gone 2016-19, back 2020-26 (+41 bps, t 2.88): unstable |
| Turn of the month (40) | calendar_effects.py | not significant |
| Time-series momentum on 16 ETFs (46) | tsmom.py | Sharpe 0.33-0.55; long/flat 0.74-0.83; small diversifier |
| Closing pressure reverses overnight (57, 74) | overnight/test.py, validate.py | t -9 on 2016-26 but nearly all from 2020; absent 2014-15; validation failed |
| Announcement-morning overnight premium (73) | overnight/test.py O2 | not significant |
| The rule on 97 stocks and 15 ETFs (breadth) | exploration | no edge before costs outside QQQ/SPY/XLK |
| Index-level half-hour seasonality (11) | batch B1 | opposite sign: reversal, t -4.10 (not the predicted continuation) |
| Last 15 minutes (91) | batch B2 | no |
| Treasury auction cycle on TLT (76) | batch B3 | right sign, t 0.63 |
| Morning sunshine in New York (69) | batch B4 | no (t +0.58) |
| HMM regimes (82) | batch B5 | no (t -0.08) |
| Bitcoin weekly momentum (77) | batch B6 | right sign, t 1.92, short data |
| RSI(5) as the source paper's gamma proxy (8) | rsi.py R1 | right sign, t -1.88 |
| Real dealer gamma from options open interest (5, 7, 66, 92, 94) | gex.py G1/G2 2017-22 | linear test failed (t -1.58/-1.85); top-third effect seen (t -2.07/-2.97, exploratory); validation on 2023-26 running |

Not testable with our data (kept for reference): order-book models (51-53), customer option flow (64),
news and LLM text (18, 19), customer-supplier links (88), FX fixings (74-75), retail flow (59-60),
crypto funding carry (78), smirk (65), Google/Wikipedia searches (72), music/sports sentiment (70-71).
