import math
import os

import mcdc
import numpy as np

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"


MATERIAL_SYMBOL = "Al"
ENERGY = 1e6  # eV
CSDA_RANGE = 0.569  # g/cm2
ANGLE = 0.0

N_PARTICLES = 1000
z0 = 0.0

RHO_G_CM3 = 2.70  # g/cm3
ATOMIC_WEIGHT_G_MOL = 26.7497084  # g/mol
AREAL_DENSITY_G_CM2 = 5.05e-3  # g/cm2

dz = AREAL_DENSITY_G_CM2 / RHO_G_CM3
AVAGADRO_NUMBER = 6.02214076e23
MAT_DENSITY_ATOMS_PER_BARN_CM = AVAGADRO_NUMBER / ATOMIC_WEIGHT_G_MOL * RHO_G_CM3 / 1e24
TINY = 1e-30
L = CSDA_RANGE / RHO_G_CM3
N_LAYERS = int(L / dz)
THETA = math.radians(ANGLE)


mat = mcdc.Material(
    element_composition={MATERIAL_SYMBOL: MAT_DENSITY_ATOMS_PER_BARN_CM}
)

s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneZ(z=L, boundary_condition="vacuum")

mcdc.Cell(region=+s1 & -s2, fill=mat)

mcdc.Source(
    z=[z0 + TINY, z0 + TINY],
    particle_type="electron",
    energy=np.array([[ENERGY - 1, ENERGY + 1], [0.5, 0.5]]),
    direction=[math.sin(THETA), 0.0 + TINY, math.cos(THETA)],
)

z_bins = np.linspace(0.0, L, N_LAYERS + 1)
mesh = mcdc.MeshStructured(z=z_bins)

mcdc.Tally(name="edep", mesh=mesh, scores=["energy_deposition"])
mcdc.Tally(name="flux", scores=["flux"], mesh=mesh)
mcdc.Tally(name="s1_current", surface=s1, scores=["net-current"])
mcdc.Tally(name="s2_current", surface=s2, scores=["net-current"])

mcdc.settings.set_transported_particles(["electron"])
mcdc.settings.set_electron_elastic_mode("gfp2")
mcdc.settings.set_electron_gfp2_policy("regime")
mcdc.settings.set_electron_gfp2_scheme("elastic_only")
mcdc.settings.N_particle = N_PARTICLES
mcdc.settings.active_bank_buffer = N_PARTICLES * 1000

mcdc.run()
