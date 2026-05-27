import mcdc
import numpy as np
import os

# Set MCDC library path
os.environ["MCDC_LIB"] = "../mcdc_data"

# =============================================================================
# Parameters
# =============================================================================
RADIUS = 10.0         # cm
E_SOURCE = 14.0e6     # eV

N_PARTICLE = 100
N_INACTIVE = 5
N_BATCH = 50

# =============================================================================
# Material
# =============================================================================
U_235 = mcdc.Material(nuclide_composition={"U235": 0.048807514}) # atom/b-cm

# =============================================================================
# Geometry
# =============================================================================
sphere = mcdc.Surface.Sphere(
    center=[0.0, 0.0, 0.0],
    radius=RADIUS,
    boundary_condition="vacuum",
)

fuel_cell = mcdc.Cell(
    region=-sphere,
    fill=U_235,
)

# =============================================================================
# Source
# =============================================================================
mcdc.Source(
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    energy=E_SOURCE,
    time=0.0,
)

# =============================================================================
# Tallies
# =============================================================================
# Energy grid: thermal + epithermal + fast
E_thermal = np.logspace(-4, 0, 20)              # 1e-4 -> 1 eV
E_epi = np.logspace(0, 5, 20)                   # 1 -> 1e5 eV
E_fast = np.logspace(5, np.log10(20e6), 30)     # 1e5 -> 20 MeV

E_axis = np.unique(np.concatenate([E_thermal, E_epi, E_fast]))

mcdc.Tally(
    name="flux_EIG",
    scores=["flux"],
    cell=fuel_cell,
    energy=E_axis,
)

# ============================================================================
# Settings
# ============================================================================
mcdc.settings.N_particle = N_PARTICLE
mcdc.settings.active_bank_buffer = 100 * N_PARTICLE
mcdc.settings.census_bank_buffer_ratio = 10.0
mcdc.settings.source_bank_buffer_ratio = 5.0

# Eigenvalue mode
mcdc.settings.set_eigenmode(
    N_inactive=N_INACTIVE,
    N_active=N_BATCH,
    gyration_radius="all",
)

# CGMF SETTING:
#mcdc.settings.set_fission_emission_model("cgmf")

# Techniques
mcdc.simulation.population_control()

# Run
mcdc.run()
