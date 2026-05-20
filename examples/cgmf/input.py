import mcdc
import numpy as np
import os

# Set MCDC library path
os.environ["MCDC_LIB"] = "/nfs/stak/users/dermanm/5-Jack/MCDC/examples/mcdc_data"

# ============================================
# Set Model
# ============================================

# Set material
U_235 = mcdc.Material(nuclide_composition={"U235": 0.048807514})

# Geometry
sphere = mcdc.Surface.Sphere(center=[0, 0, 0], radius=8.0, boundary_condition="vacuum")
inside_sphere = -sphere
sphere_cell = mcdc.Cell(region=inside_sphere, fill=U_235)

# ============================================
# Set Source (Time dependent pulse)
# ============================================
# At t=0, 14 MeV neutrons are emitted from the center
ENERGY = 14e6  # 14 MeV neutrons
mcdc.Source(
    position=[0, 0, 0],
    isotropic=True,
    energy=np.array(
        [[ENERGY - 1.0, ENERGY + 1.0], [0.5, 0.5]]
    ),  # Monoenergetic source with a small energy bin
    time=0.0,
)  # Instantaneous source at t=0

# ============================================
# Axes (For Analysis)
# ============================================

# Time Axis: 0 to 100 nanoseconds, 200 bins
# Considering the speed of neutrons and the size of the sphere,
# nanosecond scale is appropriate for this problem.
t_axis = np.linspace(0, 100e-9, 51)

# Energy Axis
E_1 = np.logspace(-4, 0, 20)  # thermal: 1e-4 -> 1 eV
E_2 = np.logspace(0, 5, 20)  # epithermal: 1 -> 1e5 eV
E_3 = np.logspace(5, np.log10(14e6), 30)  # fast: 1e5 -> 14 MeV
E_axis = np.concatenate([E_1, E_2[1:], E_3[1:]])


# ============================================
# Tallies (Time and Energy Dependent)
# ============================================
# Recording how the flux inside the sphere changes over time
mcdc.Tally(scores=["flux"], time=t_axis, energy=E_axis)

# ===========
# Settings
# ===========
N = 1e3  # For statistical significance, at least 100,000 particles
mcdc.settings.N_particle = int(N)
mcdc.settings.active_bank_buffer = int(100 * N)

# CRITICAL SETTING: To check the CGMF implementation
mcdc.settings.set_fission_emission_model("cgmf")

# Time boundary for the simulation (100 ns)
mcdc.settings.time_boundary = 100e-9

# Run
mcdc.run()
