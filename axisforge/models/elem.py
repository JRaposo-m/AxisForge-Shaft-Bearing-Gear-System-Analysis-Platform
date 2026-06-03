from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Elem:
    length: float
    E: float
    I: float
    A: float
    v: float
    idx_node_1: int
    idx_node_2: int