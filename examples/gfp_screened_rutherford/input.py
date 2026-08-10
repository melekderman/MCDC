import numpy as np

import mcdc

# ======================================================================================
# Set model
# ======================================================================================
# Leakeas-Larsen two-region monoenergetic pencil beam problem.

# Set materials
region_1 = mcdc.MaterialMG(
    capture=np.array([0.0]),
    scatter=np.array([[100.0]]),
    scatter_eta=1.0e-4,
)
region_2 = mcdc.MaterialMG(
    capture=np.array([0.0]),
    scatter=np.array([[50.0]]),
    scatter_eta=1.0e-2,
)

# Set surfaces
z0 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
zi = mcdc.Surface.PlaneZ(z=0.75)
z1 = mcdc.Surface.PlaneZ(z=3.0, boundary_condition="vacuum")

# Set cells
mcdc.Cell(region=+z0 & -zi, fill=region_1)
mcdc.Cell(region=+zi & -z1, fill=region_2)

# ======================================================================================
# Set source
# ======================================================================================

mcdc.Source(
    position=[0.0, 0.0, 1.0e-12],
    direction=[0.0, 0.0, 1.0],
    energy_group=0,
)

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Transversely integrated scalar flux: divide the output by dz.
mesh_z = mcdc.MeshStructured(z=np.linspace(0.0, 3.0, 301))
mcdc.Tally(name="transverse_integrated_flux", mesh=mesh_z, scores=["flux"])

# Cartesian window for radial profile near z = 0.8 cm.
mesh_radial = mcdc.MeshStructured(
    x=np.linspace(-0.2, 0.2, 81),
    y=np.linspace(-0.2, 0.2, 81),
    z=np.array([0.795, 0.805]),
)
mcdc.Tally(name="radial_flux_window", mesh=mesh_radial, scores=["flux"])

# Settings
mcdc.settings.N_particle = 10000
mcdc.settings.N_batch = 2
mcdc.settings.active_bank_buffer = 1000

# Run
mcdc.run()
