# We consider two important parameters in our model input: atomic mass and electronegativity.

from dataclasses import dataclass

@dataclass
class Element:
    symbol: str
    ra: float
    en: float



