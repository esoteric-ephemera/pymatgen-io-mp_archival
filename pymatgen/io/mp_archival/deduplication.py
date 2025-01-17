"""Tools for deduplicating calculations"""

from __future__ import annotations
from collections.abc import Iterator
from pydantic import BaseModel, Field

from enum import Enum
from typing import Any, TYPE_CHECKING, Optional

import numpy as np

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from pymatgen.io.mp_archival.typing import ThreeVector, ThreeByThreeMatrix

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Sequence
    from typing_extensions import Self

    from numpy.typing import ArrayLike


class ElementSymbol(Enum):
    """Lightweight representation of a chemical element."""

    H = 1
    He = 2
    Li = 3
    Be = 4
    B = 5
    C = 6
    N = 7
    O = 8
    F = 9
    Ne = 10
    Na = 11
    Mg = 12
    Al = 13
    Si = 14
    P = 15
    S = 16
    Cl = 17
    Ar = 18
    K = 19
    Ca = 20
    Sc = 21
    Ti = 22
    V = 23
    Cr = 24
    Mn = 25
    Fe = 26
    Co = 27
    Ni = 28
    Cu = 29
    Zn = 30
    Ga = 31
    Ge = 32
    As = 33
    Se = 34
    Br = 35
    Kr = 36
    Rb = 37
    Sr = 38
    Y = 39
    Zr = 40
    Nb = 41
    Mo = 42
    Tc = 43
    Ru = 44
    Rh = 45
    Pd = 46
    Ag = 47
    Cd = 48
    In = 49
    Sn = 50
    Sb = 51
    Te = 52
    I = 53
    Xe = 54
    Cs = 55
    Ba = 56
    La = 57
    Ce = 58
    Pr = 59
    Nd = 60
    Pm = 61
    Sm = 62
    Eu = 63
    Gd = 64
    Tb = 65
    Dy = 66
    Ho = 67
    Er = 68
    Tm = 69
    Yb = 70
    Lu = 71
    Hf = 72
    Ta = 73
    W = 74
    Re = 75
    Os = 76
    Ir = 77
    Pt = 78
    Au = 79
    Hg = 80
    Tl = 81
    Pb = 82
    Bi = 83
    Po = 84
    At = 85
    Rn = 86
    Fr = 87
    Ra = 88
    Ac = 89
    Th = 90
    Pa = 91
    U = 92
    Np = 93
    Pu = 94
    Am = 95
    Cm = 96
    Bk = 97
    Cf = 98
    Es = 99
    Fm = 100
    Md = 101
    No = 102
    Lr = 103
    Rf = 104
    Db = 105
    Sg = 106
    Bh = 107
    Hs = 108
    Mt = 109
    Ds = 110
    Rg = 111
    Cn = 112
    Nh = 113
    Fl = 114
    Mc = 115
    Lv = 116
    Ts = 117
    Og = 118

    @property
    def Z(self) -> int:
        return self.value

    def __str__(self):
        return self.name

class LightLattice(tuple):

    def __new__(cls, matrix):
        lattice_matrix = np.array(matrix)
        if lattice_matrix.shape != (3, 3):
            raise ValueError("Lattice matrix must be 3x3.")
        return super(LightLattice,cls).__new__(cls,tuple([tuple(v) for v in lattice_matrix.tolist()]))

    def as_dict(self) -> dict[str, list | str]:
        return {"@class": self.__class__, "@module": self.__module__, "matrix": self}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        return cls(dct["matrix"])

    def copy(self) -> Self:
        return LightLattice(self)

class LightElement(BaseModel):

    element : ElementSymbol = Field(None, description="The element.")
    lattice : Optional[ThreeByThreeMatrix] = Field(None, description="The lattice in 3x3 matrix form.")
    cart_coords : Optional[ThreeVector] = Field(None, description="The postion of the site in Cartesian coordinates.")
    frac_coords : Optional[ThreeVector] = Field(None, description="The postion of the site in direct lattice vector coordinates.")
    charge : Optional[float] = Field(None, description="The on-site charge.")
    magmom : Optional[float] = Field(None, description="The on-site magnetic moment.")

    def model_post_init(self, __context : Any) -> None:
        if self.lattice:
            if self.cart_coords is not None:
                self.frac_coords = np.linalg.solve(
                        np.array(self.lattice).T, np.array(self.cart_coords)
                    )
            elif self.frac_coords is not None:
                self.cart_coords = tuple(np.matmul(np.array(self.lattice).T, np.array(self.frac_coords)))
        
    @classmethod
    def from_element(
        cls,
        element: int | str | LightElement,
        lattice: ThreeByThreeMatrix | LightLattice | None = None,
        coords: ThreeVector | None = None,
        coords_are_cartesian: bool = False,
        **kwargs
    ):
        if isinstance(element, (int | ElementSymbol)):
            element = ElementSymbol(element)
        elif isinstance(element, str):
            element = ElementSymbol[element]
        else:
            raise ValueError(f"Unknown element {element}")

        lattice = LightLattice(lattice) if lattice else None
        cart_coords = None
        frac_coords = None
        if coords is not None:
            coords = tuple(coords)
            
        if coords_are_cartesian:
            cart_coords = coords
        else:
            frac_coords = coords

        return cls(
            element = element,
            lattice = lattice,
            cart_coords = cart_coords,
            frac_coords = frac_coords,
            **kwargs
        )

    @property
    def species(self) -> dict[str,int]:
        return {self.element.name : 1}

    @property
    def properties(self) -> dict[str,float]:
        props = {}
        for k in ("charge","magmom"):
            if (prop := getattr(self,k,None)) is not None:
                props[k] = prop
        return props

    def __int__(self) -> int:
        return self.element.Z

    def __float__(self) -> float:
        return float(self.element.Z)

    @property
    def elements(self) -> list[ElementSymbol]:
        return [self.element]

    @property
    def Z(self) -> int:
        return self.element.Z

    @property
    def name(self) -> str:
        return self.element.name
    
    @property
    def species_string(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        return self.name

    def __str__(self):
        return self.label

class LightStructure(BaseModel):
    """Light-on-memory implementation of a Structure.

    Basically a duck-typed Structure.
    """
    
    lattice : ThreeByThreeMatrix = Field(None, description="The lattice in 3x3 matrix form.")
    sites : list[LightElement] = Field(None, description="The sites in the structure.")

    def __getitem__(self, idx: int | slice) -> LightElement | list[LightElement]:
        if isinstance(idx, int) or isinstance(idx, slice):
            return self.sites[idx]
        raise IndexError("Index must be an integer or slice!")

    def __iter__(self) -> Iterator[LightElement]:
        yield from self.sites

    @property
    def frac_coords(self) -> np.ndarray:
        return np.array([self.sites[idx].frac_coords for idx in range(len(self))])

    @property
    def cart_coords(self) -> np.ndarray:
        return np.array([self.sites[idx].cart_coords for idx in range(len(self))])

    @property
    def volume(self) -> float:
        return abs(np.linalg.det(self.lattice))

    def __len__(self) -> int:
        return len(self.sites)

    @property
    def num_sites(self) -> int:
        return self.__len__()

    @classmethod
    def from_structure(cls, structure: Structure) -> Self:
        if not structure.is_ordered:
            raise ValueError(
                "Currently, `LightStructure` is intended to handle only ordered materials."
            )
        
        lattice = LightLattice(structure.lattice.matrix)
        properties = [{} for _ in range(len(structure))]
        for idx, site in enumerate(structure):
            for k in ("charge","magmom"):
                if (prop := site.properties.get(k)) is not None:
                    properties[idx][k] = prop

        sites = [
            LightElement.from_element(
                next(iter(site.species.remove_charges().as_dict())),
                lattice=lattice,
                coords=site.frac_coords,
                coords_are_cartesian=False,
                **properties[idx]
            )
            for idx, site in enumerate(structure)
        ]

        return cls(
            lattice=structure.lattice.matrix,
            sites = sites
        )

    @classmethod
    def from_poscar(cls, poscar_path: str | Path) -> Self:
        return cls.from_structure(Poscar.from_file(poscar_path).structure)

    def __str__(self):
        def _format_float(val: float | int) -> str:
            nspace = 2 if val >= 0.0 else 1
            return " " * nspace + f"{val:.8f}"

        lattice_str = [
            [_format_float(self.lattice[i][j]) for j in range(3)] for i in range(3)
        ]
        coords_str = [
            [_format_float(self.cart_coords[i][j]) for j in range(3)]
            for i in range(len(self))
        ]
        as_str = "Lattice\n"
        as_str += "\n".join(
            f"{name}  : " + ",".join(lattice_str[idx])
            for idx, name in enumerate(["a", "b", "c"])
        )
        as_str += "\nCartesian Coordinates\n"

        as_str += "\n".join(
            f"{self[idx].element}{' '*(3-len(str(self[idx].element)))}: "
            + ",".join(site_str)
            for idx, site_str in enumerate(coords_str)
        )
        return as_str
