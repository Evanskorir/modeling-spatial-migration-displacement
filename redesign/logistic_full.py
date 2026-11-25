"""Compatibility import; no return to the original 33-variable model."""
from .logistic_after_correlation import LogisticAfterCorrelation

class LogisticFull(LogisticAfterCorrelation):
    pass
