"""First fit, using correlation-retained variables before VIF screening."""
from .fractional_logit import FractionalLogit

class LogisticAfterCorrelation(FractionalLogit):
    pass
