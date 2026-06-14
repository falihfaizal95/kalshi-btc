"""paper package — virtual paper-trading account for risk-free strategy learning."""
from .account import PaperAccount, paper_trade_cycle

__all__ = ["PaperAccount", "paper_trade_cycle"]
