#!/usr/bin/python3

# spell-checker:ignore PTABLE scrf genecp Gkcal

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

from PySide6.QtCore import (
    QAbstractItemModel,
    QDir,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QRect,
    QSize,
    Qt,
    Slot,
)
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFileIconProvider,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStyledItemDelegate,
    QStyleOptionViewItem,
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


class LogParseState(Enum):
    SearchRoute = auto()
    ReadRoute = auto()
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
    # This is jank...
    if components[0].lower() in {"m06", "m062x", "b3lyp", "wb97xd"}:
        if len(components) == 1:
            return components[0], None
        return components[0], components[1]
    return None


RE_SCRF_SOLVENT = re.compile(r"Solvent=([^,)]+)", re.IGNORECASE)
RE_CHARGE_MULT = re.compile(r"^ Charge =\s+(-?\d+) Multiplicity = (\d+)$")
RE_ATOM_POSITION = re.compile(r"^ ([A-Za-z]{1,2})\s+(-?\d+\.\d*)\s+(-?\d+\.\d*)\s+(-?\d+\.\d*) $")
RE_IR_FREQ = re.compile(r"^ Frequencies --\s+(-?\d+\.\d*)(?:\s+(-?\d+\.\d*))?(?:\s+(-?\d+\.\d*))?$")
RE_FREE_ENERGY = re.compile(r"^ Sum of electronic and thermal Free Energies= \s+(-\d+\.\d+)$")


def au_to_kj_per_mol(au: float) -> float:
    return au * 2625.499639


KJ_TO_KCAL = 0.238_845


@enum.verify(enum.EnumCheck.CONTINUOUS)
class Column(IntEnum):
    Name = 0
    Success = auto()
    Formula = auto()
    Charge = auto()
    Mult = auto()
    Solvent = auto()
    Theory = auto()
    RealFreq = auto()
    ImagFreq = auto()
    GHartree = auto()
    GkJPerMol = auto()
    GkcalPerMol = auto()
    Reference = auto()

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
                return "Success"
            case self.RealFreq:
                return "# Re Vib"
            case self.ImagFreq:
                return "# Im Vib"
            case self.GHartree:
                return "G(au)"
            case self.GkJPerMol:
                return "G(kJ/mol)"
            case self.GkcalPerMol:
                return "G(kcal/mol)"
            case self.Reference:
                return "Ref"


COLUMNS = [c for c in Column]
assert COLUMNS[0] == Column.Name


@dataclass
class GaussianJob:
    path: Path
    charge: int = 0
    mult: int = 1  # 2S+1
    formula: MolecularFormula = dataclasses.field(default_factory=MolecularFormula)
    route: list[str] = dataclasses.field(default_factory=list)
    functional: str | None = None
    basis: str | None = None
    solvent: str | None = None
    num_real_freq: int | None = None
    num_imag_freq: int | None = None
    # electronic_energy_au = 0  # TODO
    free_energy_au: float | None = None
    success: bool | None = None

    def __hash__(self) -> int:
        return id(self)

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

                if line.startswith(" Error termination"):
                    self.success = False
                    state = LogParseState.Terminated
                    continue

                match state:
                    case LogParseState.SearchRoute:
                        if line.startswith(" #"):
                            self.route.append(line.removeprefix(" #").strip())
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
                                elif f_b := try_parse_functional_basis(kw):
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
                        else:
                            if any(kw.startswith("irc") for kw in self.route):
                                state = LogParseState.ReadTermination
                            else:
                                state = LogParseState.SearchIr
                    case LogParseState.SearchIr:
                        if line.startswith(" Harmonic frequencies (cm**-1)"):
                            state = LogParseState.ReadIr
                    case LogParseState.ReadIr:
                        if self.num_real_freq is None:
                            self.num_real_freq = 0
                        if self.num_imag_freq is None:
                            self.num_imag_freq = 0
                        if match := RE_IR_FREQ.match(line):
                            for g in match.groups():
                                if not g:
                                    continue
                                if float(g) < 0:
                                    self.num_imag_freq += 1
                                else:
                                    self.num_real_freq += 1
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

    def _replace(self, other: Self):
        self.__dict__.update(other.__dict__)

    def __str__(self) -> str:
        return (
            f"GaussianJob({self.formula}({self.charge:+}, 2S+1={self.mult}), "
            f"{self.functional}/{self.basis}, Solvent={self.solvent}, "
            f"G={self.free_energy_au:.4f}au)"
        )

    def data(self, col: Column, ref: "References | None" = None) -> str | int:
        g_au = self.free_energy_au
        if g_au and ref:
            tally = ref.tally()
            if tally.is_comparable(self):
                g_au -= tally.g_au
            else:
                g_au = None

        match col:
            case Column.Name:
                return self.path.stem
            case Column.Formula:
                return f"{self.formula}"
            case Column.Charge:
                return f"{self.charge:+}"
            case Column.Mult:
                return self.mult
            case Column.Solvent:
                return f"{self.solvent}"
            case Column.Theory:
                return f"{self.functional or 'unknown'}/{self.basis or 'unknown'}"
            case Column.Success:
                return f"{self.success}" if self.success is not None else "Unknown"
            case Column.RealFreq:
                return f"{self.num_real_freq}" if self.num_real_freq is not None else "N/A"
            case Column.ImagFreq:
                return f"{self.num_imag_freq}" if self.num_imag_freq is not None else "N/A"
            case Column.GHartree:
                return f"{g_au:+.6f}" if g_au else "N/A"
            case Column.GkJPerMol:
                return f"{au_to_kj_per_mol(g_au):+.4f}" if g_au else "N/A"
            case Column.GkcalPerMol:
                return f"{au_to_kj_per_mol(g_au) * KJ_TO_KCAL:+.4f}" if g_au else "N/A"
            case Column.Reference:
                return 1


@dataclass(frozen=True)
class FuturesHolder:
    futures: list[Future[GaussianJob]] = dataclasses.field(default_factory=list)
    replacers: list[Callable[[GaussianJob], None]] = dataclasses.field(default_factory=list)

    def append(self, future: Future[GaussianJob], replacer: Callable[[GaussianJob], None]):
        self.futures.append(future)
        self.replacers.append(replacer)


class SpinBoxDelegate(QStyledItemDelegate):
    """https://doc.qt.io/qtforpython-6/examples/example_widgets_itemviews_spinboxdelegate.html"""
    def __init__(self, /, parent: QObject | None = None):
        super().__init__(parent)

    def createEditor(
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> QSpinBox:
        editor = QSpinBox(parent)
        editor.setMinimum(1)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex | QPersistentModelIndex):
        assert isinstance(editor, QSpinBox)
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        if isinstance(value, int):
            editor.setValue(value)
        else:
            editor.setValue(editor.valueFromText(value))

    def setModelData(
        self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex | QPersistentModelIndex
    ):
        assert isinstance(editor, QSpinBox)
        editor.interpretText()
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(
        self,
        editor: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ):
        rect: QRect = option.rect  # type: ignore
        decoration: QSize = option.decorationSize  # type: ignore
        rect.adjust(max(decoration.width(), 0), 0, 0, 0)
        editor.setGeometry(rect)


@dataclass(frozen=True)
class ReferenceTally:
    formula: MolecularFormula
    charge: int
    g_au: float

    def is_comparable(self, job: GaussianJob) -> bool:
        return self.formula.is_empty() or self.formula == job.formula and self.charge == job.charge


class References:
    def __init__(self):
        self._inner: Counter[GaussianJob] = Counter()

    def set(self, job: GaussianJob, count: int):
        assert job.free_energy_au is not None
        assert count >= 0
        self._inner[job] = count

    def clear(self):
        self._inner.clear()

    def tally(self) -> ReferenceTally:
        formula = MolecularFormula()
        charge = 0
        g_au = 0.0
        for job, count in self._inner.items():
            for _ in range(count):
                formula += job.formula
                charge += job.charge
                g_au += job.free_energy_au or 0.0
        return ReferenceTally(formula, charge, g_au)


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

        self.dirs.sort(key=lambda dir: dir.path)
        self.jobs.sort(key=lambda job: job.path)

        duration = time.time() - start
        print(f"Parsed {self.len_recurse()} jobs in {duration:.2f} s.")

    def _scan_recurse(self, executor: Executor, holder: FuturesHolder):
        for item in self.path.iterdir():
            if item.is_dir():
                self.dirs.append(DataDirectory(item))  # type:ignore
            elif item.is_file() and item.suffix == ".log":
                job = GaussianJob(item)
                holder.append(executor.submit(job.parse), job._replace)
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

    def _dir_row(self) -> list[QStandardItem]:
        name = QStandardItem(self.path.name)
        name.setIcon(self._ICON_PROVIDER.icon(QFileIconProvider.IconType.Folder))
        name.setEditable(False)
        return [name]

    def _job_row(self, idx: int) -> tuple[GaussianJob, list[QStandardItem]]:
        job = self.jobs[idx]

        row = [QStandardItem() for _ in COLUMNS]
        for col in COLUMNS:
            row[col].setData(job.data(col), Qt.ItemDataRole.EditRole)
            row[col].setEditable(False)

        row[Column.Name].setIcon(self._ICON_PROVIDER.icon(QFileIconProvider.IconType.File))
        row[Column.Reference].setEnabled(job.free_energy_au is not None)
        row[Column.Reference].setCheckable(True)
        row[Column.Reference].setEditable(True)

        return job, row

    def build_model_recurse(
        self, parent: QStandardItem, checkbox_to_job: dict[QModelIndex, GaussianJob]
    ):
        if self.collapsible():
            job, row = self._job_row(0)
            row[Column.Name].setText(self.path.name)
            row[Column.Name].setIcon(self._ICON_PROVIDER.icon(QFileIconProvider.IconType.File))
            parent.appendRow(row)
            checkbox_to_job[row[Column.Reference].index()] = job
        else:
            this_row = self._dir_row()
            parent.appendRow(this_row)
            this_dir = this_row[Column.Name]
            for dir_ in self.dirs:
                dir_.build_model_recurse(this_dir, checkbox_to_job)
            for i in range(len(self.jobs)):
                job, row = self._job_row(i)
                this_dir.appendRow(row)
                checkbox_to_job[row[Column.Reference].index()] = job


class DataDirectoryModel(QStandardItemModel):
    def __init__(self, parent: QObject):
        super().__init__(parent)
        self.directory = DataDirectory()
        self.itemChanged.connect(self._checkbox_changed)
        self.checkbox_to_job: dict[QModelIndex, GaussianJob] = {}
        self.references = References()

    def load_path(self, path: Path):
        self.directory.path = path
        self.directory.scan_concurrently()

        self.clear()
        self.checkbox_to_job.clear()
        self.references.clear()
        self.setHorizontalHeaderLabels([str(col) for col in COLUMNS])
        self.directory.build_model_recurse(self.invisibleRootItem(), self.checkbox_to_job)

    @Slot(QStandardItem)  # type: ignore
    def _checkbox_changed(self, item: QStandardItem):
        if item.column() != Column.Reference:
            return
        reference_job = self.checkbox_to_job[item.index()]
        if item.checkState() == Qt.CheckState.Checked:
            self.references.set(reference_job, item.data(Qt.ItemDataRole.EditRole))
        else:
            self.references.set(reference_job, 0)
        self._rerender_energies()

    def clear_references(self):
        for checkbox_idx in self.checkbox_to_job:
            checkbox = self.itemFromIndex(checkbox_idx)
            if not checkbox.isCheckable():
                continue
            checkbox.setCheckState(Qt.CheckState.Unchecked)

    def _rerender_energies(self):
        for checkbox_idx, job in self.checkbox_to_job.items():
            for col in [Column.GHartree, Column.GkJPerMol, Column.GkcalPerMol]:
                item = self.itemFromIndex(checkbox_idx.siblingAtColumn(col))
                if self.itemFromIndex(checkbox_idx).checkState() == Qt.CheckState.Checked:
                    item.setText("0 (ref)")
                else:
                    item.setData(job.data(col, self.references), Qt.ItemDataRole.EditRole)


class GaussScreen(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("GaussScreen")

        main_layout = QVBoxLayout(self)

        self.box_path = QGroupBox("Active path")
        self.textbox_path = QLineEdit(self.box_path)
        self.box_path.setLayout(QHBoxLayout())
        self.box_path.layout().addWidget(self.textbox_path)  # type: ignore
        self.btn_browser = QPushButton("Browse", self.box_path)
        self.btn_browser.clicked.connect(self.select_active_path)
        self.box_path.layout().addWidget(self.btn_browser)  # type: ignore
        self.btn_reload = QPushButton("Reload", self.box_path)
        self.btn_reload.clicked.connect(self.reload_path)
        self.box_path.layout().addWidget(self.btn_reload)  # type: ignore
        main_layout.addWidget(self.box_path)

        self.box_opt = QGroupBox("Options")
        self.box_opt.setLayout(QHBoxLayout())
        self.btn_clear_ref = QPushButton("Clear References", self.box_opt)
        self.btn_clear_ref.clicked.connect(self.clear_references)
        self.box_opt.layout().addWidget(self.btn_clear_ref)  # type: ignore
        main_layout.addWidget(self.box_opt)

        self.model = DataDirectoryModel(self)
        self.tree = QTreeView(self)
        self.tree.setItemDelegateForColumn(Column.Reference, SpinBoxDelegate())
        self.tree.setModel(self.model)
        main_layout.addWidget(self.tree)

        self.setLayout(main_layout)

        self.select_active_path()

    @Slot()
    def select_active_path(self):
        dialog = QFileDialog(self)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if self.textbox_path.text():
            dialog.setDirectory(self.textbox_path.text())
        if dialog.exec():
            self.textbox_path.setText(dialog.selectedFiles()[0])
        elif not self.textbox_path.text():
            self.textbox_path.setText(QDir.currentPath())
        self.reload_path()

    @Slot()
    def reload_path(self):
        self.model.load_path(Path(self.textbox_path.text()))
        self.tree.expandAll()
        for i in range(self.model.columnCount() - 1):
            self.tree.resizeColumnToContents(i)

    @Slot()
    def clear_references(self):
        self.model.clear_references()


if __name__ == "__main__":
    app = QApplication([])

    g_screen = GaussScreen()
    g_screen.resize(1200, 800)
    g_screen.show()

    sys.exit(app.exec())
