"""Second fit, using the VIF-retained variables."""
from .fractional_logit import FractionalLogit

class LogisticRetained(FractionalLogit):
    """No additional significance-based variable removal."""
    pass
