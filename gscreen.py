import concurrent.futures
import dataclasses
import enum
import textwrap
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from enum import IntEnum, auto
from pathlib import Path
from typing import Self

from PySide6.QtCore import QModelIndex, QObject, Qt, Slot
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
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

from gaussian_job import GaussianJob
from reference import References
from spin_box import SpinBoxDelegate
from utils import KJ_TO_KCAL, au_to_kj_per_mol


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
    EHartree = auto()
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
            case self.EHartree:
                return "E(au)"
            case self.GHartree:
                return "G(au)"
            case self.GkJPerMol:
                return "G(kJ/mol)"
            case self.GkcalPerMol:
                return "G(kcal/mol)"
            case self.Reference:
                return "Ref"

    def extract_from(self, job: GaussianJob, ref: References | None = None) -> str | int:
        # TODO: reference EE too.
        g_au = job.free_energy_au
        if g_au is not None and ref:
            tally = ref.tally()
            if tally.is_comparable(job):
                g_au -= tally.g_au
            else:
                g_au = None

        match self:
            case Column.Name:
                return job.path.stem
            case Column.Formula:
                return f"{job.formula}"
            case Column.Charge:
                return f"{job.charge:+}"
            case Column.Mult:
                return job.mult
            case Column.Solvent:
                return f"{job.solvent}"
            case Column.Theory:
                return f"{job.functional or 'unknown'}/{job.basis or 'unknown'}"
            case Column.Success:
                return f"{job.success}" if job.success is not None else "Unknown"
            case Column.RealFreq:
                return n if (n := job.num_real_freqs()) is not None else "N/A"
            case Column.ImagFreq:
                return n if (n := job.num_imag_freqs()) is not None else "N/A"
            case Column.EHartree:
                return f"{ee_au:+.6f}" if (ee_au := job.elec_energy_au()) is not None else "N/A"
            case Column.GHartree:
                return f"{g_au:+.6f}" if g_au is not None else "N/A"
            case Column.GkJPerMol:
                return f"{au_to_kj_per_mol(g_au):+.4f}" if g_au is not None else "N/A"
            case Column.GkcalPerMol:
                return f"{au_to_kj_per_mol(g_au) * KJ_TO_KCAL:+.4f}" if g_au is not None else "N/A"
            case Column.Reference:
                return 1


COLUMNS = [c for c in Column]
assert COLUMNS[0] == Column.Name


@dataclasses.dataclass(frozen=True)
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

    def try_collapse(self) -> Self | None:
        if not self.dirs and len(self.jobs) == 1:
            return self
        elif len(self.dirs) == 1:
            return self.dirs[0].try_collapse()
        return None

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
            row[col].setData(col.extract_from(job), Qt.ItemDataRole.EditRole)
            row[col].setEditable(False)

        row[Column.Name].setIcon(self._ICON_PROVIDER.icon(QFileIconProvider.IconType.File))
        row[Column.Reference].setEnabled(job.free_energy_au is not None)
        row[Column.Reference].setCheckable(True)
        row[Column.Reference].setEditable(True)

        return job, row

    def build_model_recurse(
        self, parent: QStandardItem, checkbox_to_job: dict[QModelIndex, GaussianJob]
    ):
        if (inner := self.try_collapse()) is not None:
            (job, row) = inner._job_row(0)
            row[Column.Name].setText(str(job.path.parent.relative_to(self.path.parent)))
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
                    item.setData(col.extract_from(job, self.references), Qt.ItemDataRole.EditRole)


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
        self.textbox_path.setText(QFileDialog.getExistingDirectory(self))
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
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication([])

    g_screen = GaussScreen()
    g_screen.resize(1200, 800)
    g_screen.show()

    sys.exit(app.exec())
