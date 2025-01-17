"""Define types used in models."""
from __future__ import annotations

ThreeVector = tuple[float, float, float]
#ThreeVector.__doc__ = "Three-component vector of floats."

ThreeByThreeMatrix = tuple[ThreeVector,ThreeVector,ThreeVector]
#ThreeByThreeMatrix.__doc__ = "3x3 matrix of floats."

ThreeByThreeVoigt = tuple[float, float, float,float, float, float]
#ThreeByThreeVoigt.__doc__ = "3x3 symmetric matrix of floats in Voigt representation."

Force = tuple[ThreeVector,...]
#Force.__doc__ = "Sequence of three-vectors corresponding to sites within a structure."