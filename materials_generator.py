# materials_generator.py
# Python 3.12 – Generator für "neue Materialien" (Perowskit, Spinell, Heusler, Legierung).

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import math, random

OX = {
    "Na": +1, "K": +1, "Rb": +1, "Cs": +1,
    "Ca": +2, "Sr": +2, "Ba": +2,
    "La": +3, "Y": +3,
    "Ti": +4, "V": +3, "Cr": +3, "Mn": +4, "Fe": +3, "Co": +3, "Ni": +2, "Cu": +2, "Zr": +4, "Hf": +4,
    "Al": +3, "Ga": +3,
    "Sn": +4, "Sb": +3, "Si": +4, "Ge": +4,
}

RAD = {
    "Na": 1.18, "K": 1.51, "Rb": 1.61, "Cs": 1.74,
    "Ca": 1.12, "Sr": 1.26, "Ba": 1.42,
    "La": 1.36, "Y": 1.22,
    "Ti": 0.74, "V": 0.78, "Cr": 0.80, "Mn": 0.83, "Fe": 0.78, "Co": 0.74, "Ni": 0.69, "Cu": 0.73, "Zr": 0.86, "Hf": 0.83,
    "Al": 0.67, "Ga": 0.62, "Sn": 0.81, "Sb": 0.76, "Si": 0.40, "Ge": 0.53,
    "O": 1.35,
}

EN = {
    "Na": 0.93, "K": 0.82, "Rb": 0.82, "Cs": 0.79,
    "Ca": 1.00, "Sr": 0.95, "Ba": 0.89,
    "La": 1.10, "Y": 1.22,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zr": 1.33, "Hf": 1.30,
    "Al": 1.61, "Ga": 1.81, "Sn": 1.96, "Sb": 2.05, "Si": 1.90, "Ge": 2.01,
    "O": 3.44,
}

A_CAND = ["Na","K","Rb","Cs","Ca","Sr","Ba","La","Y"]
B_CAND = ["Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zr","Hf","Al","Ga"]
X_CAND = ["Ni","Co","Fe","Mn","Cu"]
Y_CAND = ["Ti","V","Cr","Ga","Al"]
Z_CAND = ["Sn","Sb","Si","Ge"]

@dataclass
class NewMaterial:
    formula: str
    proto: Literal["perovskite","spinel","heusler","alloy"]
    params: dict
    x: list[float]

def perovskite_candidates(n: int, rng: random.Random):
    res = []
    for _ in range(n * 4):
        A = rng.choice(A_CAND); B = rng.choice(B_CAND)
        scheme = rng.choice([(2,4), (3,3)])
        qA, qB = scheme
        rA = RAD.get(A,1.2); rB = RAD.get(B,0.7); rO = RAD["O"]
        t = (rA + rO) / (math.sqrt(2)*(rB + rO))
        if 0.8 <= t <= 1.05:
            en_contrast = abs((EN.get(A,1.0)+EN["O"])/2 - EN.get(B,1.5))
            formula = f"{A}{'2' if qA==2 else '3'}{B}{'4' if qB==4 else '3'}O3"
            x = [t, en_contrast, rA, rB, float(qA), float(qB)]
            res.append(NewMaterial(formula, "perovskite", {"A":A,"B":B,"charges":(qA,qB),"t":t}, x))
            if len(res) >= n: break
    return res[:n]

def spinel_candidates(n: int, rng: random.Random):
    res = []
    for _ in range(n * 6):
        A = rng.choice(A_CAND); B = rng.choice(B_CAND)
        qA, qB = 2, 3
        rA = RAD.get(A,1.1); rB = RAD.get(B,0.7)
        if rA <= rB: continue
        en_contrast = abs(EN.get(B,1.5) - EN["O"])
        formula = f"{A}1{B}2O4"
        x = [rA, rB, en_contrast, float(qA), float(qB)]
        res.append(NewMaterial(formula, "spinel", {"A":A,"B":B,"charges":(qA,qB)}, x))
        if len(res) >= n: break
    return res

def heusler_candidates(n: int, rng: random.Random):
    res = []
    for _ in range(n * 8):
        X = rng.choice(X_CAND); Y = rng.choice(Y_CAND); Z = rng.choice(Z_CAND)
        if EN.get(X,1.7) <= EN.get(Z,1.8) and EN.get(Y,1.7) <= EN.get(Z,1.8):
            formula = f"{X}2{Y}{Z}"
            x = [EN[X], EN[Y], EN[Z], RAD.get(X,0.8), RAD.get(Y,0.8), RAD.get(Z,0.8)]
            res.append(NewMaterial(formula, "heusler", {"X":X,"Y":Y,"Z":Z}, x))
            if len(res) >= n: break
    return res

def alloy_candidates(n: int, rng: random.Random):
    res = []
    metals = list({*X_CAND, *Y_CAND})
    for _ in range(n * 6):
        A, B = rng.sample(metals, 2)
        xfrac = rng.uniform(0.05, 0.95)
        en_gap = abs(EN.get(A,1.6) - EN.get(B,1.6))
        r_gap  = abs(RAD.get(A,0.8) - RAD.get(B,0.8))
        score_proxy = 1.0 / (1e-6 + 0.5*en_gap + 0.5*r_gap)
        formula = f"{A}{1.0-xfrac:.2f}{B}{xfrac:.2f}"
        res.append(NewMaterial(formula, "alloy", {"A":A,"B":B,"x":xfrac}, [xfrac, en_gap, r_gap, score_proxy]))
        if len(res) >= n: break
    return res

def propose_new_materials(n: int, seed: int = 7, mix: tuple[int,int,int,int] = (4,4,3,5)):
    rng = random.Random(seed)
    pa, ps, ph, pl = mix
    cand = []
    cand += perovskite_candidates(pa, rng)
    cand += spinel_candidates(ps, rng)
    cand += heusler_candidates(ph, rng)
    cand += alloy_candidates(pl, rng)
    rng.shuffle(cand)
    return cand[:n]
