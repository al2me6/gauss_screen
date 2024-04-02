#!/usr/bin/python3

# spell-checker:ignore PTABLE scrf genecp

import concurrent.futures
import dataclasses
import enum
import mmap
import re
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Self

from PySide6.QtCore import QDir, Slot
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFileIconProvider,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


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

    def len(self) -> int:
        return len(self.symbols)


PTABLE = PeriodicTable()


class MolecularFormula:
    def __init__(self) -> None:
        self._data: Counter[int] = Counter()

    def add_atom(self, sym: str):
        if sym in PTABLE.atomic_nums:
            self._data[PTABLE.atomic_nums[sym]] += 1

    def __repr__(self) -> str:
        ret = ""
        for z in PTABLE.iter_order:
            if z in self._data:
                ret += PTABLE.symbols[z]
                if self._data[z] > 1:
                    ret += f"{self._data[z]}"
        return ret


class LogParseState(Enum):
    SearchRoute = auto()
    SearchStructure = auto()
    ReadChargeMult = auto()
    ReadFormula = auto()
    SearchIr = auto()
    ReadIr = auto()
    ReadFreeEnergy = auto()
    ReadTermination = auto()
    Terminated = auto()


def try_parse_functional_basis(keyword: str) -> tuple[str, str | None] | None:
    components = keyword.split("/", maxsplit=1)
    if components[0].lower() in {"m06", "m062x", "b3lyp"}:
        if len(components) == 1:
            return components[0], None
        return components[0], components[1]
    return None


RE_SCRF_SOLVENT = re.compile(r"Solvent=([^,)]+)", re.IGNORECASE)
RE_CHARGE_MULT = re.compile(r"^ Charge =\s+(-?\d+) Multiplicity = (\d+)$")
RE_ATOM_POSITION = re.compile(r"^ ([A-Za-z]{1,2})\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+) $")
RE_IR_FREQ = re.compile(r"^ Frequencies --\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?(?:\s+(-?\d+\.\d+))?$")
RE_FREE_ENERGY = re.compile(r"^ Sum of electronic and thermal Free Energies= \s+(-\d+\.\d+)$")


def hartree_to_kj_per_mol(au: float) -> float:
    return au * 2625.499639


@enum.verify(enum.EnumCheck.CONTINUOUS)
class Column(IntEnum):
    Name = 0
    Success = auto()
    Formula = auto()
    Charge = auto()
    Mult = auto()
    Solvent = auto()
    Theory = auto()
    ImagFreq = auto()
    GHartree = auto()
    GkJPerMol = auto()

    def __str__(self):
        match self:
            case self.Name:
                return "Name"
            case self.Formula:
                return "Formula"
            case self.Charge:
                return "Charge"
            case self.Mult:
                return "2S+1"
            case self.Solvent:
                return "Solvent"
            case self.Theory:
                return "Theory"
            case self.Success:
                return "Succeeded"
            case self.ImagFreq:
                return "# Imag Freq"
            case self.GHartree:
                return "G(au)"
            case self.GkJPerMol:
                return "G(kJ/mol)"


COLUMNS = [c for c in Column]


@dataclass
class GaussianJob:
    path: Path
    charge: int = 0
    mult: int = 1  # 2S+1
    formula: MolecularFormula = dataclasses.field(default_factory=MolecularFormula)
    functional: str = ""
    basis: str = ""
    solvent: str | None = None
    num_imag_freq: int = 0
    # electronic_energy_au = 0  # TODO
    free_energy_au = 0
    success: bool = False

    def parse(self):
        print(f"Parsing path {self.path}...")

        with open(self.path, "r+b") as log_file:
            # ew
            try:
                log = mmap.mmap(log_file.fileno(), 0)
            except ValueError:
                log = mmap.mmap(log_file.fileno(), 1)

            state = LogParseState.SearchRoute
            while state != LogParseState.Terminated:
                line = log.readline().decode()
                if not line:
                    state = LogParseState.Terminated
                    continue
                line = line.strip("\n")

                match state:
                    case LogParseState.SearchRoute:
                        if not line.startswith(" #"):
                            continue
                        keywords = line.removeprefix(" #").split()
                        for kw in keywords:
                            kw_lower = kw.lower()
                            if kw_lower == "genecp":
                                self.basis = kw
                            elif kw_lower.startswith("scrf"):
                                if match := RE_SCRF_SOLVENT.search(kw):
                                    self.solvent = match.group(1)
                            elif f_b := try_parse_functional_basis(kw):
                                self.functional = f_b[0]
                                if f_b[1]:
                                    self.basis = f_b[1]
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
                        else:
                            state = LogParseState.SearchIr
                    case LogParseState.SearchIr:
                        if line.startswith(" Harmonic frequencies (cm**-1)"):
                            state = LogParseState.ReadIr
                    case LogParseState.ReadIr:
                        if match := RE_IR_FREQ.match(line):
                            for g in match.groups():
                                if float(g) < 0:
                                    self.num_imag_freq += 1
                        if not line:
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

    def replace(self, other: Self):
        self.__dict__.update(other.__dict__)

    def __str__(self) -> str:
        return f"GaussianJob({self.formula}({self.charge:+}, 2S+1={self.mult}), {self.functional}/{self.basis}, Solvent={self.solvent}, G={self.free_energy_au:.4f}au)"

    def fmt(self, col: Column) -> str:
        match col:
            case Column.Name:
                return self.path.stem
            case Column.Formula:
                return f"{self.formula}"
            case Column.Charge:
                return f"{self.charge:+}"
            case Column.Mult:
                return f"{self.mult}"
            case Column.Solvent:
                return f"{self.solvent}"
            case Column.Theory:
                return f"{self.functional}/{self.basis}"
            case Column.Success:
                return f"{self.success}"
            case Column.ImagFreq:
                return f"{self.num_imag_freq}"
            case Column.GHartree:
                return f"{self.free_energy_au:.6f}"
            case Column.GkJPerMol:
                return f"{hartree_to_kj_per_mol(self.free_energy_au):.4f}"


@dataclass(frozen=True)
class FuturesHolder:
    futures: list[Future[GaussianJob]] = dataclasses.field(default_factory=list)
    replacers: list[Callable[[GaussianJob], None]] = dataclasses.field(default_factory=list)

    def append(self, future: Future[GaussianJob], replacer: Callable[[GaussianJob], None]):
        self.futures.append(future)
        self.replacers.append(replacer)


class DataDirectory:
    _ICON_PROVIDER = QFileIconProvider()

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path()
        self.dirs: list[Self] = []
        self.jobs: list[GaussianJob] = []

    def scan_concurrently(self):
        start = time.time()

        with ProcessPoolExecutor() as executor:
            holder = FuturesHolder()

            self.dirs.clear()
            self.jobs.clear()
            self._scan_recurse(executor, holder)

            _, not_done = concurrent.futures.wait(holder.futures)
            assert not not_done
            for res, replace in zip(holder.futures, holder.replacers, strict=True):
                if exc := res.exception():
                    raise exc
                replace(res.result())

        self._prune_empty()

        duration = time.time() - start
        print(f"Parsed {self.len_recurse()} jobs in {duration:.2f} s.")

    def _scan_recurse(self, executor: Executor, holder: FuturesHolder):
        for item in self.path.iterdir():
            if item.is_dir():
                self.dirs.append(DataDirectory(item))  # type:ignore
            elif item.is_file() and item.suffix == ".log":
                job = GaussianJob(item)
                holder.append(executor.submit(job.parse), job.replace)
                self.jobs.append(job)
        for subdir in self.dirs:
            subdir._scan_recurse(executor, holder)

    def _contains_job_recurse(self) -> bool:
        if self.jobs:
            return True
        for dir_ in self.dirs:
            if dir_._contains_job_recurse():
                return True
        return False

    def _prune_empty(self):
        for i, dir_ in enumerate(self.dirs):
            if not dir_._contains_job_recurse():
                del self.dirs[i]
            else:
                dir_._prune_empty()

    def __str__(self) -> str:
        ret = f"{self.path.name}\n"
        for dir_ in self.dirs:
            ret += textwrap.indent(str(dir_), " " * 4)
        for job in self.jobs:
            ret += f"  {job}\n"
        return ret

    def collapsible(self) -> bool:
        return not self.dirs and len(self.jobs) == 1

    def len_recurse(self) -> int:
        return len(self.jobs) + sum(dir_.len_recurse() for dir_ in self.dirs)

    def dir_row_(self) -> list[QStandardItem]:
        name = QStandardItem(self.path.name)
        name.setIcon(self._ICON_PROVIDER.icon(QFileIconProvider.IconType.Folder))
        name.setEditable(False)
        return [name]

    def job_row_(self, idx: int) -> list[QStandardItem]:
        job = self.jobs[idx]

        row = [QStandardItem(job.fmt(col)) for col in COLUMNS]
        for item in row:
            item.setEditable(False)

        row[Column.Name].setIcon(self._ICON_PROVIDER.icon(QFileIconProvider.IconType.File))

        return row

    def collapsed_row_(self) -> list[QStandardItem]:
        row = self.job_row_(0)
        name = QStandardItem(self.path.name)
        name.setEditable(False)
        name.setIcon(self._ICON_PROVIDER.icon(QFileIconProvider.IconType.File))
        row[Column.Name] = name
        return row

    def build_model_recurse_(self, parent: QStandardItem):
        this_row = self.collapsed_row_() if self.collapsible() else self.dir_row_()
        this_dir = this_row[Column.Name]

        parent.appendRow(this_row)

        if not self.collapsible():
            for dir_ in self.dirs:
                dir_.build_model_recurse_(this_dir)
            for i in range(len(self.jobs)):
                this_dir.appendRow(self.job_row_(i))

    def rebuild_model(self, model: QStandardItemModel):
        model.clear()
        model.setHorizontalHeaderLabels([str(col) for col in COLUMNS])
        self.build_model_recurse_(model.invisibleRootItem())


class GaussScreen(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("GaussScreen")

        main_layout = QVBoxLayout(self)

        self.active_path_box = QGroupBox("Active path")
        self.path = QLineEdit(self.active_path_box)
        self.active_path_box.setLayout(QHBoxLayout())
        self.active_path_box.layout().addWidget(self.path)
        self.browse_path = QPushButton("Browse", self.active_path_box)
        self.browse_path.clicked.connect(self.select_active_path)
        self.active_path_box.layout().addWidget(self.browse_path)
        main_layout.addWidget(self.active_path_box)

        self.filters_box = QGroupBox("Filters")
        self.filters_box.setLayout(QVBoxLayout())
        self.filter_only_success = QCheckBox("Show only success", self.filters_box)
        self.filters_box.layout().addWidget(self.filter_only_success)
        main_layout.addWidget(self.filters_box)

        self.directory = DataDirectory()
        self.model = QStandardItemModel(self)
        self.tree = QTreeView(self)
        self.tree.setModel(self.model)
        main_layout.addWidget(self.tree)

        self.setLayout(main_layout)

        # self.select_active_path()  # REMOVE BEFORE FLIGHT
        self.path.setText(QDir.currentPath())
        self.directory.path = Path(self.path.text())
        self.refresh_tree()

    @Slot()
    def select_active_path(self):
        dialog = QFileDialog(self)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if self.path.text():
            dialog.setDirectory(self.path.text())
        if dialog.exec():
            self.path.setText(dialog.selectedFiles()[0])
        elif not self.path.text():
            self.path.setText(QDir.currentPath())
        self.directory.path = Path(self.path.text())
        self.refresh_tree()

    def refresh_tree(self):
        self.directory.scan_concurrently()
        self.directory.rebuild_model(self.model)
        self.model.layoutChanged.emit()
        self.tree.expandAll()
        for i in range(self.model.columnCount() - 1):
            self.tree.resizeColumnToContents(i)


if __name__ == "__main__":
    app = QApplication([])

    g_screen = GaussScreen()
    g_screen.resize(1200, 800)
    g_screen.show()

    sys.exit(app.exec())
