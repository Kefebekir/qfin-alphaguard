# Reference data

## `sp500_membership.csv`

S&P 500 membership over time: one row per continuous stretch in the index. A
ticker is a member from `start_date` up to, but not including, `end_date`; an
empty `end_date` means still a member on the latest snapshot (18 August 2026).

- **Source:** `sp500_ticker_start_end.csv` from
  [fja05680/sp500](https://github.com/fja05680/sp500), commit
  `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`, copied on 8 October 2026 without
  changes.
- **Check:** on every one of the 2,720 snapshot dates in the same commit's
  `S&P 500 Historical Components & Changes (Updated).csv`, the members this file
  gives match the snapshot exactly.
- **Updating:** copy the file from a newer commit, update the commit and
  snapshot date above, and run the tests. Changes to the index after the latest
  snapshot are unknown until then.

### Licence

MIT License

Copyright (c) 2019-2020 Farrell J. Aultman <fja0568@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
