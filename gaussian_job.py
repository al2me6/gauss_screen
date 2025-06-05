import dataclasses
import re
from enum import Enum, auto
from pathlib import Path
from typing import Self

from atom_coords import AtomCoords
from formula import PTABLE, MolecularFormula

RE_SCRF_SOLVENT = re.compile(r"Solvent=([^,)]+)", re.IGNORECASE)
RE_CHARGE_MULT = re.compile(r"^ Charge =\s+(-?\d+) Multiplicity =\s+(\d+)$")
RE_ATOM_POSITION = re.compile(r"^ ([A-Za-z]{1,2})\s+(-?\d+\.\d*)\s+(-?\d+\.\d*)\s+(-?\d+\.\d*)\s?$")
# Index, atomic number, "atomic type" (?), x, y, z.
RE_ATOM_COORDS = re.compile(r"^\s+\d+\s+(\d+)\s+\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)$")
RE_ELEC_ENERGY = re.compile(r"^ SCF Done:  E\(.+\) =\s+(-?\d+\.\d*)\s+A.U. after\s+\d+ cycles")
RE_IR_FREQ = re.compile(r"^ Frequencies --\s+(-?\d+\.\d*)(?:\s+(-?\d+\.\d*))?(?:\s+(-?\d+\.\d*))?$")
RE_FREE_ENERGY = re.compile(r"^ Sum of electronic and thermal Free Energies= \s+(-\d+\.\d+)$")


class LogParseState(Enum):
    SearchRoute = auto()
    ReadRoute = auto()
    SearchStructure = auto()
    ReadChargeMult = auto()
    ReadFormula = auto()
    SearchGeometry = auto()
    ReadGeometry = auto()
    SearchEE = auto()
    SearchIr = auto()
    ReadIr = auto()
    ReadFreeEnergy = auto()
    ReadTermination = auto()
    Terminated = auto()


@dataclasses.dataclass
class GaussianJob:
    path: Path
    charge: int = 0
    mult: int = 1  # 2S+1
    formula: MolecularFormula = dataclasses.field(default_factory=MolecularFormula)
    route: list[str] = dataclasses.field(default_factory=list)
    functional: str | None = None
    basis: str | None = None
    solvent: str | None = None
    vibrations: list[float] | None = None
    _elec_energy_au: list[float] | None = None
    free_energy_au: float | None = None
    coordinates: list[list[AtomCoords]] | None = None
    success: bool | None = None

    def __hash__(self) -> int:
        return id(self)

    def elec_energy_au(self) -> float | None:
        if not self._elec_energy_au:
            return None
        return self._elec_energy_au[-1]

    def num_real_freqs(self) -> int | None:
        if not self.vibrations:
            return None
        return len([nu for nu in self.vibrations if nu > 0])

    def num_imag_freqs(self) -> int | None:
        if not self.vibrations:
            return None
        return len([nu for nu in self.vibrations if nu < 0])

    @staticmethod
    def _try_parse_functional_basis(keyword: str) -> tuple[str, str | None] | None:
        components = keyword.split("/", maxsplit=1)
        # This is jank...
        if components[0].lower() in {"m06", "m062x", "b3lyp", "wb97xd"}:
            if len(components) == 1:
                return components[0], None
            return components[0], components[1]
        return None

    def parse(self):
        print(f"Parsing path {self.path}...")

        with open(self.path, "rb") as log_file:
            # ew
            # try:
            #     log = mmap.mmap(log_file.fileno(), 0)
            # except ValueError:
            #     log = mmap.mmap(log_file.fileno(), 1)
            log = log_file

            state = LogParseState.SearchRoute
            while state != LogParseState.Terminated:
                line = log.readline().decode()

                if not line:
                    state = LogParseState.Terminated
                    continue
                line = line.rstrip("\n")

                if line.startswith(" Error termination"):
                    self.success = False
                    state = LogParseState.Terminated
                    continue

                match state:
                    case LogParseState.SearchRoute:
                        if line.startswith(" #"):
                            self.route.append(line.removeprefix(" #"))
                            state = LogParseState.ReadRoute
                    case LogParseState.ReadRoute:
                        if set(line.strip()) != {"-"}:
                            # Remove only the first character, which is always a space.
                            self.route[0] += line[1:]
                        else:
                            self.route = [kw.lower().strip() for kw in self.route[0].split()]
                            for kw in self.route:
                                if kw == "genecp":
                                    self.basis = kw
                                elif kw.startswith("scrf"):
                                    if match := RE_SCRF_SOLVENT.search(kw):
                                        self.solvent = match.group(1)
                                elif f_b := GaussianJob._try_parse_functional_basis(kw):
                                    (self.functional, self.basis) = f_b
                            state = LogParseState.SearchStructure
                    case LogParseState.SearchStructure:
                        if line == " Symbolic Z-matrix:":
                            state = LogParseState.ReadChargeMult
                    case LogParseState.ReadChargeMult:
                        match = RE_CHARGE_MULT.match(line)
                        assert match is not None
                        self.charge = int(match.group(1))
                        self.mult = int(match.group(2))
                        state = LogParseState.ReadFormula
                    case LogParseState.ReadFormula:
                        if match := RE_ATOM_POSITION.match(line):
                            self.formula.add_atom(match.group(1))
                        elif any(kw.startswith("irc") for kw in self.route):
                            state = LogParseState.ReadTermination
                        else:
                            state = LogParseState.SearchGeometry
                    case LogParseState.SearchGeometry:
                        if line.startswith(" Optimization complete"):
                            state = LogParseState.SearchIr
                        elif line.strip() == "Standard orientation:":
                            # Skip header.
                            for _ in range(4):
                                log.readline()
                            if not self.coordinates:
                                self.coordinates = []
                            self.coordinates.append([])
                            state = LogParseState.ReadGeometry
                    case LogParseState.ReadGeometry:
                        assert self.coordinates is not None
                        if match := RE_ATOM_COORDS.match(line):
                            self.coordinates[-1].append(
                                AtomCoords(
                                    PTABLE.element(int(match.group(1))),
                                    *(float(match.group(i)) for i in [2, 3, 4]),
                                )
                            )
                        else:
                            state = LogParseState.SearchEE
                    case LogParseState.SearchEE:
                        if match := RE_ELEC_ENERGY.match(line):
                            if not self._elec_energy_au:
                                self._elec_energy_au = []
                            self._elec_energy_au.append(float(match.group(1)))
                            state = LogParseState.SearchGeometry
                    case LogParseState.SearchIr:
                        if line.startswith(" Harmonic frequencies (cm**-1)"):
                            self.vibrations = []
                            state = LogParseState.ReadIr
                    case LogParseState.ReadIr:
                        assert self.vibrations is not None
                        if match := RE_IR_FREQ.match(line):
                            for g in match.groups():
                                if g:
                                    self.vibrations.append(float(g))
                        if not line:
                            self.vibrations.sort()
                            state = LogParseState.ReadFreeEnergy
                    case LogParseState.ReadFreeEnergy:
                        if match := RE_FREE_ENERGY.match(line):
                            self.free_energy_au = float(match.group(1))
                            state = LogParseState.ReadTermination
                    case LogParseState.ReadTermination:
                        if line.startswith(" Normal termination of Gaussian"):
                            self.success = True
                            state = LogParseState.Terminated
        return self

    def _replace(self, other: Self):
        self.__dict__.update(other.__dict__)

    def __str__(self) -> str:
        return (
            f"GaussianJob({self.formula}({self.charge:+}, 2S+1={self.mult}), "
            f"{self.functional}/{self.basis}, Solvent={self.solvent}, "
            f"G={self.free_energy_au:.4f}au)"
        )
