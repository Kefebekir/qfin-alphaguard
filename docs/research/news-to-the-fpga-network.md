# News, and the network that will carry it to the FPGA

Research before N2, written on 18 October 2026. Efe's question: the news models will one day
feed the neural network on the FPGA, so the architecture must be right before N2 starts. This
note gathers what our own data say, what the literature says, what follows for N2, and an
architecture from text to a network the FPGA can run. Nothing here changes a pre-registered
test: the choices it raises are Efe's.

## The short answer

- **Text never goes on the FPGA.** A language model that reads news has 150 million (ChronoBERT)
  to 4 billion (PIT-4B) parameters; the planned board, an Arty A7-100T, has 240 DSP slices.
  News arrives in seconds and matters for hours or days, so microseconds buy nothing there.
  What crosses to the fast side is a small, fixed, versioned **news state**: a few numbers a
  stock, updated when an article arrives, written to the journal like any other input.
- **The network that goes on the FPGA is small and quantized from its first training run.**
  An MLP of a few thousand 8-bit weights, taking the news state and market features, fits the
  board and answers in well under a microsecond. Its worth is judged in the shadow record
  (decision 0020) in Python first; the FPGA runs the same integer arithmetic, bit for bit.
- **For N2, the evidence is sobering.** In our development years, the next-day continuation
  after news among S&P 500 members is about **+4 bps a day** before costs (t 0.83), against
  N2's pre-registered cost of **20 bps a day**. The literature's large daily effects come from
  equal-weighted portfolios of all stocks, mostly small ones. N2 as written will very likely
  fail on costs, whatever the model reads.
- **Costs are the lever N2 got wrong, not the model.** N2's 10 bps a side was a placeholder.
  Alpaca's market-on-close orders fill at the official close, with no spread; for an account
  our size the cost is close to the regulatory fees, about 0.2 bps a sale (the SEC fee our
  cost model uses). Whether
  Alpaca takes such orders from our account is not settled: its documentation marks them
  "contact the sales team". A paper test on a second paper account settles it.
- **Bigger point-in-time models help, and they exist.** Kelly, Malamud, Schwab and Xu (NeurIPS
  2026) trained 4-billion-parameter models only on text up to each date; monthly portfolios
  from their news embeddings reach Sharpe ratios of 1.0-1.5, against 0.67 for ChronoGPT in
  the same test. Their checkpoints are public (Apache-2.0), but need care: they run their own
  code on loading, and come from an anonymous account.
- **News says little about size in our data.** A member's news count adds almost nothing to
  its next day's move beyond its recent volatility (R² +0.0005), and the morning's count adds
  little to QQQ's range (R² +0.0065 over 0.404). If news is to help the QQQ rule's sizing, it
  will be through what the articles say, not how many there are, and the shadow record must
  show it.

## 1. What our own work already shows

| Finding | Where |
| --- | --- |
| Prices tell how much the market will move, not which way: R² ≤ 0 for direction, 15-18% for size | `research/frontier/results/01_diagnosis.txt` |
| After an intraday move with news, the move carries on (slope +0.065, t 4.7, 98 stocks); without news it reverses | `research/frontier/results/02_news_reaction.txt` |
| N1 failed: the morning's headlines, read by ChronoBERT and a linear model, did not tell the QQQ rule's day (−1.52 bps a day, t −1.75) | `research/news_reader/LEDGER.md` |
| N2's count-only comparison loses 25 bps a day on development, of which 20 are costs | `research/news_reader/LEDGER.md` |

New measurements for this note, on the development years only (2016-2021), so N2's holdout
is untouched (`research/news_reader/architecture_checks.py`,
`results_architecture_checks.txt`):

| Question | Result |
| --- | --- |
| Among members with news in N2's window, the day's best tenth less its worst tenth: the next day, before costs | **+4.02 bps a day**, t 0.83, 1,495 days. Without news: −6.44, t −1.63 (reversal). An upper bound: day t's return runs 30 minutes past N2's window |
| Does the news count tell the next day's absolute move, beyond 20-day volatility? | Volatility alone R² 0.171; log(1 + count) adds **+0.0005**. Members with news move 0.858 of their volatility the next day, without news 0.809 |
| Does the morning's count tell QQQ's range, beyond 14-day volatility? | 14-day volatility R² 0.404; the count adds **+0.0065** |

## 2. What the literature says

**News and next-day returns are real, but live in small stocks and the first day.**

- ChronoBERT's own paper (He, Lv, Manela and Wu): daily long-short deciles from Dow Jones news
  embeddings, January 2008 - July 2023, **equal-weighted across all CRSP stocks with news**:
  61% a year before costs, Sharpe 4.80, about 24 bps a day. Most of it is earned on the first
  day; predictability falls steeply over two days. No transaction costs are deducted.
- Lopez-Lira and Tang: GPT-4 scores predict the drift after headlines, **especially for small
  stocks and bad news**, and strategy returns fall as LLM use spreads.
- Jiang, Li and Wang (JFE 2021): prices underreact to firm news and drift for several days;
  the drift is stronger when investors are distracted and, in the companion "News
  Momentum", for small firms and overnight and weekend news.
- So for S&P 500 members, our measured +4 bps a day is in line: large caps absorb news fast.

**Lower turnover, residualized embeddings and scale work better than daily trading.**

- Kelly, Malamud, Schwab and Xu, "Scaling Point-in-Time Language Models" (NeurIPS 2026):
  - decoder models of 1.5 and 4 billion parameters, trained on 1 trillion tokens of FineWeb
    filtered by date, with yearly checkpoints for 2013-2024;
  - news embeddings averaged by month and stock, and **residualized against known firm
    characteristics**;
  - combined by a penalized maximum-Sharpe regression;
  - out of sample, 2014 - May 2020, Sharpe ratios: ChronoGPT 0.67, PIT-4B 1.02, the
    fine-tuned PIT-4B 1.53. Scale is most of the gap to models trained without time limits.
- The ChronoGPT-Instruct paper estimates that at least 54-62% of the apparent predictability
  of larger models survives once leakage is removed.

**Look-ahead and distraction.** Glasserman and Lin show that models trained on later text can
know what followed a headline, and that what a model knows about a company can bias its
reading of it (a "distraction effect"). Point-in-time models solve the first by construction.
The second is worth one variant: headlines with company names masked.

**Costs at the close.**
- Goyal, Jegadeesh and Wu (2025) find the closing auction carries about 10% of daily volume,
  with lower price impact than continuous trading for all but Nasdaq micro caps.
- For an account of 30,000 USD, a market-on-close order fills at the official close: no
  spread, no measurable impact, only the regulatory fees on sales (about 0.2 bps, the SEC
  fee in our cost model).
- Alpaca documents market- and limit-on-close orders (`cls`) for whole shares only, sent by
  15:50 New York time, and marks them "contact the sales team". Fractional orders cannot use
  them, and shorts are whole shares anyway.

## 3. What that means for N2

N2, as approved on 16 October:
- members with news to 15:30;
- ChronoBERT of the year before;
- Efe's PCA and ridge;
- long the top tenth and short the bottom tenth, close to close;
- **10 bps a round trip on each side**, so 20 bps a day;
- pass at t ≥ 2.24 on 2022 - 9 October 2026.

To pass, the model must earn about 20 bps a day before costs on S&P 500 members: five times
the bound we measured from prices alone, and close to what ChronoBERT's paper found for all
stocks, small ones included. The test as written measures costs more than it measures news.

Three honest paths, for Efe to choose:

1. **Run N2 exactly as written.** Nothing changes; the report includes the before-cost
   result, which says whether the model finds anything at all. The likely verdict is "failed".
2. **Amendment 2, costs only, before any model result.**
   - First verify on a second paper account that Alpaca takes `cls` orders from us. Then
     set the cost to what such an order costs (fees, about 0.2 bps a sale, rounded up to
     1 bp a side a round trip).
   - If `cls` is refused, use the measured cost of a marketable order in the last minute:
     our NBBO measurements put the half spread at 15:59 near 1.7 bps for the members that
     gap most, and about 0.3 bps for QQQ (`docs/research/intraday-machine-learning.md`).
   - The amendment must say that the count-only development check (−25 bps, of which 20
     costs) had been seen, and that the new cost comes from the measurement, not from that
     result.
   - Everything else stays as approved.
3. **Keep N2 and pre-register N3 before N2's holdout runs**, a design the evidence favours:
   - PIT-4B embeddings (December of the year before, or the month before);
   - residualized against momentum, reversal, size and volatility, which our prices give;
   - held a week, with a fifth rebalanced each day;
   - costs as measured.

   It shares N2's holdout years, so the threshold tightens (Bonferroni over N1, N2 and N3:
   t ≥ about 2.39).

My recommendation is 2 and 3 together: the cost was an assumption, the project's rule is
measured costs, and N3 tests the design the literature points to. But it is Efe's call, and
path 1 is the purest.

## 4. The architecture: from text to a network an FPGA can run

Two speeds, joined by one narrow, versioned interface.

```
 SLOW SIDE (CPU/GPU, seconds)                 │  FAST SIDE (CPU now, FPGA later, microseconds)
                                              │
 Alpaca news stream ──► text encoder ──► embedding (768-4,096 numbers)
   (Benzinga)           ChronoBERT /          │
                        PIT-4B, vintage       │
                        before the year       │
                              │               │
                              ▼               │
                      news state: frozen      │      market features (returns, volatility,
                      projection to 16,  ─────┼────► band and VWAP distance, time of day)
                      decayed by half-life,   │                   │
                      int8, per stock and     │                   ▼
                      market-wide; journaled  │      decision network: small MLP, int8,
                      with its known-at time  │      trained quantized; a fixed-point
                                              │      golden model in Python and C++
                                              │                   │
                                              │                   ▼
                                              │      a challenger card in the shadow record
                                              │      (size multiplier or score) ──► Guard
```

### 4.1 The text encoder

- **One rule, as in N1/N2:** text of year Y is read by a model that knows nothing after Y−1
  (ChronoBERT yearly vintages; PIT-4B December checkpoints, or the month before).
- **Embeddings are cached by article id with the time the article became known.** An edited
  article counts from its edit, as `embed.py` already does. Nothing downstream may use an
  embedding before that time.
- **It runs where it is cheap:**
  - ChronoBERT on a CPU handles a day's 600-900 live articles easily;
  - PIT-4B needs a GPU: Efe's 8 GB card in 8-bit, or a GPU instance for a nightly batch.
- **It is never on the FPGA.** Transformer inference on FPGAs is real (hls4ml runs small
  transformers under 2 µs, and about 100 ns for a quantized jet tagger), but those models
  have thousands of parameters, not hundreds of millions, and run on boards a hundred times
  larger than ours.

### 4.2 The news state: the only thing that crosses

- **A projection fitted on training years only and frozen for the test year** (principal
  components, or a learned linear layer), from the embedding to k = 16 numbers. It is
  versioned with the encoder's vintage.
- **Per stock and market-wide:**
  - an exponentially decayed sum of the projected articles, with a half-life fixed in
    advance (a day for the daily horizon, an hour intraday);
  - log(1 + articles) and the minutes since the last one.
- **Quantized to int8 with scales fixed on training years,** so the fast side receives exactly
  the integers the FPGA will.
- **Written to the journal as an input with its known-at time,** so a replayed day is the same
  day, and no decision can see an article before it arrived.

### 4.3 The decision network

- **Inputs:** the news state (16 + 3 numbers) and market features (about 16), all in fixed
  point.
- **Shape:** an MLP such as 35 → 32 → 16 → 1, about 1,700 multiply-adds, small enough to try
  many seeds, and to fit the board.
- **Trained quantized from the start (quantization-aware training):**
  - Brevitas in PyTorch, which Efe already trains with (`research/gpu_sim/efe_net.py`);
  - or HGQ, which learns a bit width for each weight and has cut FPGA resources by up to 20
    times in published work;
  - a model trained in floating point and rounded later is a different model.
- **A golden model:** the same integer arithmetic in numpy, and later in C++, gives the
  network's output bit for bit. The FPGA must match it (Phase 5's "golden model" test).
- **Its output is a challenger, never an order.**
  - For the QQQ rule: a size multiplier, as G2 was framed (the network learns how much to
    hold, not which way).
  - For members: a score, as N2's model.
  - It enters the shadow record on a frozen card. Guard is unchanged and stays the
    authority.

### 4.4 The FPGA

- **Toolchain:** Brevitas (PyTorch, quantization-aware) → QONNX → hls4ml (Vitis HLS, any
  AMD part including Artix-7) → Vivado → bitstream.
  - FINN is the alternative, aimed at Zynq and Alveo boards. A 2025 study fitted quantized
    MLPs with it on the smallest Artix-7 (XC7A15T).
- **Size on the Arty A7-100T,** an estimate for Phase 5 to replace with measurement:
  - 1,700 8-bit multiply-adds with a reuse factor of about 8 use under 240 DSP slices;
  - at 100 MHz, three layers take about 40 cycles, 0.4 µs;
  - HGQ with distributed arithmetic moves multipliers into look-up tables and shrinks this
    further.
- **Verification:**
  - a cocotb testbench feeds recorded news states and features, and compares the outputs
    with the golden model bit for bit;
  - latency is counted in cycles on the board, apart from the PC ↔ board transfer
    (ARCHITECTURE.md's benchmark rules).
- **What it proves and what it does not.**
  - It proves the model runs deterministically in hardware, in an independent failure
    domain, at a measured latency: the engineering the project is built to show.
  - It does not make trading through Alpaca faster or better: an order's round trip is
    30-300 ms (ARCHITECTURE.md).
  - Whether the model earns anything is the shadow record's answer, in Python, before any
    hardware.
- **Benchmarks for scale:**
  - STAC-ML's audited LSTM inference on FPGAs (Myrtle.ai VOLLO) reached about 5 µs at the
    99th percentile in 2023, on Intel Agilex cards, which a 2026 audit claims to halve on
    AMD Versal;
  - an AWS F2 FPGA instance costs about 2 USD an hour (0.70 spot), if the Arty is ever too
    small.

## 5. What to test, in order

1. **N2:** run as written, or with amendment 2 (Efe's choice, section 3). Efe trains, with at
   most 10 variants on 2018-2021, then the holdout once.
2. **N3,** if Efe wants it: pre-registered before N2's holdout runs, as in section 3.
3. **The news state** offline (the projection, the decay, the int8 scales), then live from
   Alpaca's news stream into the journal; parity tested between the batch and live
   computations (the "feature parity" test of the roadmap).
4. **G2,** pre-registered: the QQQ rule's size from market features and the news state,
   trained quantized by Efe, on a card in the shadow record. Our measurements say the news
   count adds little; the test is whether the articles' content does.
5. **Phase 5's network:** the network that earned its card, ported to the board, bit for bit
   against the golden model, its latency measured.

## 6. Risks and costs

- **PIT-4B:**
  - It loads with `trust_remote_code`, running code from an anonymous Hugging Face account:
    the code must be read and a commit pinned before use.
  - Its weights are stored in 32-bit, about 16 GB a checkpoint, so eleven yearly checkpoints
    are near 180 GB to download.
  - The model cards show the same "snapshot month" for every checkpoint. Each cutoff should
    be probed on events after it, as ChronoGPT's authors did.
- **Benzinga** covers large caps well, but is one source. Edited articles count from their
  edit (2.5%).
- **Alpaca's closing orders** are whole shares, sent by 15:50, possibly not for our account.
  A test needs a second paper account, so the live engine's reconciliation never meets an
  order it did not make.
- **Overfitting:**
  - every test is pre-registered, with variant budgets, and judged once on its holdout and
    then forward in the shadow record;
  - the networks' seeds and settings are ledgered, as in G1.
- **The FPGA work** is valued as engineering (ARCHITECTURE.md). It must never become the
  reason to believe a model earns money.

## Sources

- He, Lv, Manela and Wu, Chronologically Consistent Large Language Models:
  https://arxiv.org/abs/2502.21206
- He, Lv, Manela and Wu, Instruction Tuning Chronologically Consistent Language Models:
  https://arxiv.org/abs/2510.11677
- Kelly, Malamud, Schwab and Xu, Scaling Point-in-Time Language Models (NeurIPS 2026):
  https://arxiv.org/abs/2607.11889; checkpoints https://huggingface.co/anonymneurips2027/PIT-4B-202412
- Lopez-Lira and Tang, Can ChatGPT Forecast Stock Price Movements?: https://arxiv.org/abs/2304.07619
- Glasserman and Lin, Assessing Look-Ahead Bias in Stock Return Predictions Generated by GPT
  Sentiment Analysis: https://arxiv.org/abs/2309.17322
- Jiang, Li and Wang, Pervasive Underreaction: Evidence from High-Frequency Data, JFE 2021:
  https://papers.ssrn.com/abstract=2679614
- Boudoukh, Feldman, Kogan and Richardson, Information, Trading, and Volatility: Evidence from
  Firm-Specific News, RFS 2019
- Goyal, Jegadeesh and Wu, closing auctions and trading costs (2025), via
  https://jfqa.org/wp-content/uploads/2026/01/25551_Price_Impact.pdf
- NYSE, Closing auction: market impact, price drift and cost (2023):
  https://www.nyse.com/data-insights-blog/closing-auction-immediate-market-impact-price-drift-and-transaction-cost-of-trading-part-2-eqjcqvixj8
- Alpaca, Orders at Alpaca (time in force `cls`, `opg`): https://docs.alpaca.markets/docs/orders-at-alpaca
- Alpaca, News API and stream: https://alpaca.markets/docs/news-api
- Jiang et al., Low Latency Transformer Inference on FPGAs for Physics Applications with
  hls4ml: https://arxiv.org/abs/2409.05207
- Sub-microsecond Transformers for Jet Tagging on FPGAs (NeurIPS 2025): https://arxiv.org/abs/2510.24784
- da4ml, Distributed Arithmetic for Real-time Neural Networks on FPGAs: https://arxiv.org/abs/2507.04535;
  HGQ2: https://pypi.org/project/HGQ2/
- Learning Quantized Continuous Controllers for Integer Hardware (FINN on XC7A15T):
  https://arxiv.org/abs/2511.07046
- STAC-ML Tacana Suite on Myrtle.ai VOLLO: https://stacresearch.com/node/47622
- AWS EC2 f2.6xlarge pricing: https://instances.vantage.sh/aws/ec2/f2.6xlarge
