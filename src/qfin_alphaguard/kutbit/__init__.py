"""KUTBIT, Efe's own intraday model with leverage (decision 0023).

It is separate from the QQQ rule: nothing in momentum.py or the live QQQ strategy imports
it, and it trades from accounts of its own. This package holds what training and live
trading must share: `features`, what KUTBIT sees of a session at each minute. Its
simulator, decision layer and strategy follow in their own pull requests.
"""
