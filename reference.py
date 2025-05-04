from collections import Counter
from dataclasses import dataclass

from formula import MolecularFormula
from gaussian_job import GaussianJob


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
                g_au += job.free_energy_au if job.free_energy_au is not None else 0.0
        return ReferenceTally(formula, charge, g_au)
