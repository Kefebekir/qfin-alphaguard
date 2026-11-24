# KUTBIT's network, step by step

This is design A of decision 0027: the ladder's third rung and variant 2 of the ten. The code
is one file, `research/kutbit/net.py`, in seven steps, each with its tests in
`tests/test_kutbit_net.py`. The harness that feeds and judges it is `net_harness.py`, and its
rows are the skill test's (`net_data.py`).

PyTorch is a tool for tensors and derivatives only (decision 0024, section 7). The layers,
the loss, Adam and the learning loop are written in `net.py`; there is no `torch.nn` and no
`torch.optim`, and a test checks it.

## The idea in five lines

1. For each horizon of 1, 5, 15, 30 and 60 minutes, the network forecasts `z`: a stock's or
   an ETF's move apart from QQQ, in its usual moves. It is the same target, from the same 971
   inputs on the same rows, that the ridge forecast in the skill test.
2. A linear path does what the ridge does. Beside it, two hidden layers add what a straight
   line misses. A pattern might hold on wild days but not calm ones, or only when the market
   moved too: a straight line can only weigh each input the same way everywhere.
3. It starts by forecasting nothing: the linear path and the hidden path's last layer are
   zero. Whatever it forecasts later, it learned.
4. It learns by the mean square error, with Adam, in batches of 4,096 rows, from a seed. Five
   seeds learn it, and its forecast is their mean.
5. The harness chooses its size and how long it learns on 2019, measures it once on
   2020-2021, and asks it to beat the ridge and the same network on shuffled labels, day by
   day, at a t of 2.

## Where it runs

PyTorch lives in `data/llmenv`, beside the project's environment, with the GPU. Run every
test from the repository's root:

```bash
PYTHONPATH=src data/llmenv/bin/python -m pytest tests/test_kutbit_net.py -k step3
```

In the project's environment and in CI these tests are skipped, since PyTorch is not there.

## Step 1: the constants

Decision 0027 fixed them before anything ran:
- `BATCH`: 4,096 rows in each step of learning.
- `RATE`: each step of Adam moves a number by about 0.001.
- `BETAS`: 0.9 and 0.999, how long Adam remembers a gradient and its square. These are
  Adam's usual values (Kingma and Ba, 2015).
- `DECAYED`: the weights that weight decay pulls toward zero. The biases are left out: a bias
  only moves a forecast up or down, so shrinking it buys no simplicity.

Tests: `-k step1`. They also check that `net.py` uses no ready-made layer or optimiser.

## Step 2: the parameters (`init`)

Seven tensors, for 971 inputs, 5 horizons and a hidden width W of 128 or 512:

| Name | Shape | Starts at |
| --- | --- | --- |
| `linear` | 971 × 5 | zero |
| `bias` | 5 | zero |
| `w1`, `b1` | 971 × W, W | He's scale, zero |
| `w2`, `b2` | W × W, W | He's scale, zero |
| `w3` | W × 5 | zero |

- **He's scale.** A hidden layer's weights are drawn from a normal distribution with a
  standard deviation of √(2 / its inputs) (He, Zhang, Ren and Sun, 2015). A ReLU zeroes about
  half of what passes through it. The 2 makes up for that half, so a signal neither fades
  nor grows from layer to layer.
- **Why start at zero.** With `linear` and `w3` at zero, a new network forecasts nothing.
  The hidden layers still learn. The first gradient reaches `w3`, since what the hidden
  layers give is not zero. Once `w3` has moved, gradients flow back to `w1` and `w2`. Had
  every weight started at zero, all hidden units would get the same gradient and stay alike
  forever. So `w1` and `w2` are drawn.
- **The seed.** All the drawing comes from one generator seeded by the seed, so a seed always
  gives the same network.

Tests: `-k step2`.

## Step 3: the forecast (`forward`)

```
hidden = ReLU(x · w1 + b1)
hidden = ReLU(hidden · w2 + b2)
forecast = x · linear + bias + hidden · w3
```

- `x` is a batch of rows, 971 standardised inputs each. The harness standardises them on the
  rows the network learns from.
- A ReLU keeps a positive number and turns a negative one into zero. That bend is what lets
  the hidden layers draw a curve. Two ReLUs make |x|, which no straight line can.
- The linear path and the hidden path are added. If the hidden layers find nothing, the
  network is a linear model like the ridge. If they find something, it is on top of the line.

Tests: `-k step3`. A new network forecasts zero, a small network matches the formula worked by
hand, and a hidden unit below zero adds nothing.

## Step 4: the loss (`loss`)

The mean of the squared errors, over the batch's rows and the five horizons:

```
loss = mean((forecast − z)²)
```

- `z` is clipped at ±5 usual moves, as the ridge's was, so a freak minute cannot pull the
  network around.
- The forecast that makes the squared error smallest is the expected `z`, the edge. At its
  best scale, a forecast's squared error is the variance of `z` times (1 − its
  correlation²): the lower the error, the higher the correlation, which the gates measure.
- PyTorch computes the derivatives of the loss for every parameter. A test checks them
  against small differences, (loss(p + h) − loss(p − h)) / 2h, on a small network in double
  precision.

Tests: `-k step4`.

## Step 5: Adam, by hand (`adam`, `step`)

For each number p, with its gradient g from the last backward pass, at step t:

```
m = 0.9 · m + 0.1 · g             the running mean of the gradient
v = 0.999 · v + 0.001 · g²         the running mean of its square
m̂ = m / (1 − 0.9^t)                both start at zero, so early on they are too small;
v̂ = v / (1 − 0.999^t)              this corrects that
p = p · (1 − RATE · λ)             a weight shrinks a little first (not a bias)
p = p − RATE · m̂ / (√v̂ + 1e-8)     then moves
```

- **Why Adam.** Each number's step is divided by the usual size of its own gradient, so a
  step is about RATE at most, however large or small the gradients are. A number whose
  gradient keeps one sign moves about RATE each step; one whose gradient flips back and
  forth barely moves. On the very first step m̂/√v̂ is ±1, so every number moves by exactly
  RATE, and a test checks it.
- **Weight decay, decoupled** (Loshchilov and Hutter, 2019). λ is 0.1 or 0.01, chosen on
  2019. A weight shrinks by RATE × λ each step, apart from Adam's step. With weak signals,
  pulling weights in keeps the network from memorising noise. This is the network's
  counterpart of the ridge's penalty.
- After each step the gradients are cleared, or the next backward pass would add to them.

Tests: `-k step5`. Two steps are worked by hand with the formulas above; the first step moves
by the rate; and with no gradient only the weights shrink.

## Step 6: learning (`learn`)

```
learn(rows, width, decay, seed, epochs, after_epoch)
```

1. One generator, seeded by `seed`: it draws the weights (step 2), then each epoch's order.
2. Each epoch:
   - every row comes once, in a new random order, in batches of 4,096;
   - each batch is forecast, the loss computed, the gradients found by `backward()`, and Adam
     takes a step.
3. After each epoch it calls `after_epoch(epoch, params)`. The harness uses it to forecast
   2019 after every epoch, so the epoch is chosen there and never on 2020-2021.

`rows.batch(index)` gives the batch's standardised inputs and its `z`, on the GPU. The
harness gathers each row's 120-minute window there, so nothing is copied 120 times.

Tests: `-k step6`.
- Every row comes once an epoch, in a new order, and the last, smaller batch is not dropped.
- A seed gives the same network twice; another seed, another.
- It learns a straight line, and it learns |x|, where the best straight line explains nothing
  (R² below 0.05 for the line, above 0.8 for the network).
- Weight decay pulls the weights in.

## Step 7: the forecast, once learned (`forecast`)

The same formula as step 3, but without keeping what a gradient would need: forecasting 2019
and 2020-2021 then needs less memory and time.

Tests: `-k step7`.

## With the harness

The last test runs the harness's `smoke` and its whole placebo with this network, on a small
made-up data set: the choice, the ridge fitted again and checked against the skill test, the
network and its shuffled twin learning again, every measure, and the gates. It shows the two
fit together.

## The runs, in order

Each run starts only once the one before it has passed; the harness refuses otherwise. Each
keeps the SHA-256 of `net.py` and the harness, so a change to either means starting again from
the placebo. The times are at most, from the speed check on this laptop's GPU.

1. `smoke`, under a minute: one epoch of random-walk rows at each width, timed with Adam. It
   shows no score.
2. `placebo`, about 2 hours: the choice and the measurement on random walks. Nothing may
   reach |t| 3. If something does, the harness is wrong, and nothing runs on real prices.
3. `choose`, about 1 hour: the setting and the epoch, on 2019.
4. `measure`, about 1 hour, once: 2020-2021, the gates, and the page.

```bash
OMP_NUM_THREADS=8 PYTHONPATH=src data/llmenv/bin/python research/kutbit/net_harness.py smoke
OMP_NUM_THREADS=8 PYTHONPATH=src data/llmenv/bin/python research/kutbit/net_harness.py placebo
OMP_NUM_THREADS=8 PYTHONPATH=src data/llmenv/bin/python research/kutbit/net_harness.py choose
OMP_NUM_THREADS=8 PYTHONPATH=src data/llmenv/bin/python research/kutbit/net_harness.py measure
```

The page is `data/kutbit/net/net.html`.

These times turned out short. On 23 November the placebo took 2 h 29 min, the choice 1 h 41
min and the measurement 30 minutes, at a width of 128 for 10 epochs. The check of speed times
a stand-in network that takes no Adam step, and Adam's step, written by hand, adds about 1 ms
a batch. Decision 0027's result records what the runs found.

## In a minute

A ridge regression on 971 inputs already forecasts the next 1 to 60 minutes of a stock's move
apart from the market, a little: a correlation of about 0.01, at a t of up to 17. This network
asks whether a curve sees more in the same inputs than a straight line does. It is the
straight line plus two layers of ReLU, starts at zero, learns with Adam and weight decay
written by hand, and is averaged over five seeds. Its size, its decay and how long it learns
are chosen on 2019. It is measured once on 2020-2021 against the ridge, against itself
trained on shuffled labels, and against what a trade really costs, by rules written before
it ran.
