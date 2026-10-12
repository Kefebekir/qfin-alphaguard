# 0002: Develop, paper trade and start live on Interactive Brokers

- **Status:** replaced by decision 0008 (Alpaca, fractional shares)
- **Date:** 2026-10-07

## Context

The account is a UK retail account in GBP trading US stocks. Commission-free
brokers still charge for currency conversion, and their APIs differ a lot.
Checked in October 2026:

| Broker | US stock commission | FX cost (GBP account) | Live orders by API | Live prices by API | Paper / demo |
| --- | --- | --- | --- | --- | --- |
| IBKR | Tiered 0.0005–0.0035 USD per share, min 0.35 USD per order, plus exchange fees; Fixed 0.005 USD, min 1 USD | 0.03%; USD can be held | full, including limit orders | yes, with a subscription | yes |
| Trading 212 | 0 | 0.15%; the API trades only in the account's main currency | market orders only (API in beta) | no endpoint | demo environment |
| eToro | 0 | 0.75% on GBP to USD | market, market-if-touched, limit IOC | WebSocket | yes |

For a 1,000 USD order this is roughly 0.35–0.40 USD on IBKR Tiered, 1 USD on
IBKR Fixed, 1.50 USD on Trading 212 with a GBP account and 7.50 USD on eToro.

## Decision

Use IBKR for development, paper trading and the first live period, inside an
ISA if its API works there (to be confirmed when the account is opened).

## Consequences

- Limit orders and live prices come from one API; IB Gateway must run on an
  always-on machine, with IBC handling login and daily restarts.
- The paper account is tied to a live account, so the live account is opened
  early.
- Trading 212 becomes worth reconsidering only if its account can use USD as
  the main currency and market-only orders are acceptable.
