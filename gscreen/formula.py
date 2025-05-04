from collections import Counter
from typing import LiteralString, Self


class PeriodicTable:
    def __init__(self) -> None:
        self.symbols = (
            "H,He,"
            "Li,Be,B,C,N,O,F,Ne,"
            "Na,Mg,Al,Si,P,S,Cl,Ar,"
            "K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,"
            "Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,"
            "Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,"
            "Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,"
            "Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,"
            "Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Nh,Fl,Mc,Lv,Ts,Og"
        ).split(",")
        self.atomic_nums = dict((sym, z) for (z, sym) in enumerate(self.symbols))
        sort_first = [5, 0]  # C, H
        self.iter_order = sort_first
        self.iter_order.extend(z for z in range(len(self.symbols)) if z not in sort_first)

    def element(self, z: int) -> LiteralString:
        return self.symbols[z + 1]


PTABLE = PeriodicTable()


class MolecularFormula:
    def __init__(self) -> None:
        self._data: Counter[int] = Counter()

    def add_atom(self, sym: str):
        if sym in PTABLE.atomic_nums:
            self._data[PTABLE.atomic_nums[sym]] += 1  # type: ignore

    def clear(self):
        self._data.clear()

    def is_empty(self) -> bool:
        return self._data.total() == 0

    def __iadd__(self, rhs: Self) -> Self:
        self._data += rhs._data
        return self

    def __isub__(self, rhs: Self) -> Self:
        self._data -= rhs._data
        return self

    def __eq__(self, rhs: object) -> bool:
        if not isinstance(rhs, MolecularFormula):
            raise NotImplementedError
        return self._data == rhs._data

    def __repr__(self) -> str:
        ret = ""
        for z in PTABLE.iter_order:
            if z in self._data:
                ret += PTABLE.symbols[z]
                if self._data[z] > 1:
                    ret += f"{self._data[z]}"
        return ret
