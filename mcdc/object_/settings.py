from typing import List
import h5py
from h5py._hl.dataset import sel
import numpy as np

from dataclasses import dataclass, field
from numpy.typing import NDArray

####

from mcdc.constant import *
from mcdc.object_.base import ObjectSingleton
from mcdc.object_.util import is_sorted
from mcdc.print_ import print_error

# ======================================================================================
# Settings
# ======================================================================================


@dataclass
class Settings(ObjectSingleton):
    # Annotations for Numba mode
    label: str = "settings"

    # Basic
    N_particle: int = 0
    N_batch: int = 1
    rng_seed: int = 1

    # Simulation mode
    multigroup_mode: bool = False
    eigenvalue_mode: bool = False

    # k-eigenvalue
    N_inactive: int = 0
    N_active: int = 0
    N_cycle: int = 0
    k_init: float = 1.0
    use_gyration_radius: bool = False
    gyration_radius_type: int = GYRATION_RADIUS_ALL

    # Particle source
    use_source_file: bool = False
    source_file_name: str = ""

    # Misc.
    time_boundary: float = np.inf
    output_name: str = "output"
    use_progress_bar: bool = True

    # Time census
    N_census: int = 1
    census_time: NDArray[np.float64] = field(default_factory=lambda: np.array([np.inf]))
    use_census_based_tally: bool = False
    census_tally_frequency: int = 0

    # Particle bank-related
    save_particle: bool = False
    active_bank_buffer: int = 100
    census_bank_buffer_ratio: float = 2.0
    source_bank_buffer_ratio: float = 2.0
    future_bank_buffer_ratio: float = 1.5

    # Multi-particle options
    neutron_transport: bool = True
    electron_transport: bool = False
    proton_transport: bool = False

    # Electron elastic scattering mode
    electron_elastic_mode: int = ELECTRON_ELASTIC_MODE_COUPLED
    electron_gfp2_policy: int = ELECTRON_GFP2_POLICY_PURE
    electron_gfp2_scheme: int = ELECTRON_GFP2_SCHEME_KERNEL

    def __post_init__(self):
        super().__init__()

    def set_time_census(self, time, tally_frequency=None):
        # Make sure that the time grid points are sorted
        if not is_sorted(time):
            print_error("Time census: Time grid points have to be sorted.")

        # Make sure that the starting point is larger than zero
        if time[0] <= 0.0:
            print_error("Time census: First census time should be larger than zero.")

        # Add the default, final census-at-infinity
        time = np.append(time, np.inf)

        # Set the time census parameters
        self.census_time = time
        self.N_census = len(self.census_time)

        # Set the census-based tallying
        if tally_frequency is not None and tally_frequency > 0:
            # Flag to reset all tallies' time grids (done in main.py)
            self.use_census_based_tally = True
            self.census_tally_frequency = tally_frequency

    def set_eigenmode(
        self,
        N_inactive=0,
        N_active=0,
        k_init=1.0,
        gyration_radius=None,
        save_particle=False,
    ):
        # Update setting self
        self.N_inactive = N_inactive
        self.N_active = N_active
        self.N_cycle = self.N_inactive + self.N_active
        self.eigenvalue_mode = True
        self.k_init = k_init
        self.save_particle = save_particle

        # Gyration radius setup
        if gyration_radius is not None:
            self.use_gyration_radius = True
            if gyration_radius == "all":
                self.gyration_radius_type = GYRATION_RADIUS_ALL
            elif gyration_radius == "infinite-x":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_X
            elif gyration_radius == "infinite-y":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_Y
            elif gyration_radius == "infinite-z":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_Z
            elif gyration_radius == "only-x":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_X
            elif gyration_radius == "only-y":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_Y
            elif gyration_radius == "only-z":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_Z
            else:
                print_error("Unknown gyration radius type")

        # Allocate cycle-wise quantities
        from mcdc.object_.simulation import simulation

        simulation.k_cycle = np.zeros(self.N_cycle)
        simulation.gyration_radius = np.zeros(self.N_cycle)

    def set_source_file(self, source_file_name):
        self.use_source_file = True
        self.source_file_name = source_file_name

        # Set number of particles
        with h5py.File(source_file_name, "r") as f:
            self.N_particle = int(f["particles_size"][()])

    def set_transported_particles(self, transported_particles: List[str]):
        # Reset the flags
        self.neutron_transport = False
        self.electron_transport = False
        self.proton_transport = False

        # Set flags
        for particle in transported_particles:
            if particle == "neutron":
                self.neutron_transport = True
            elif particle == "electron":
                self.electron_transport = True
            elif particle == "proton":
                self.proton_transport = True
            else:
                print_error(r"Unsupported particle types: {particle}")

    def set_electron_elastic_mode(self, mode):
        if isinstance(mode, str):
            mode_key = mode.strip().lower()
            mapping = {
                "decoupled": ELECTRON_ELASTIC_MODE_DECOUPLED,
                "coupled": ELECTRON_ELASTIC_MODE_COUPLED,
                "gfp2": ELECTRON_ELASTIC_MODE_GFP2,
            }
            if mode_key not in mapping:
                print_error("Unknown electron elastic mode")
            mode = mapping[mode_key]

        valid_modes = {
            ELECTRON_ELASTIC_MODE_DECOUPLED,
            ELECTRON_ELASTIC_MODE_COUPLED,
            ELECTRON_ELASTIC_MODE_GFP2,
        }
        if mode not in valid_modes:
            print_error("Unknown electron elastic mode")

        self.electron_elastic_mode = mode

    def set_electron_gfp2_policy(self, policy):
        if isinstance(policy, str):
            policy_key = policy.strip().lower()
            mapping = {
                "pure": ELECTRON_GFP2_POLICY_PURE,
                "regime": ELECTRON_GFP2_POLICY_REGIME,
            }
            if policy_key not in mapping:
                print_error("Unknown electron GFP2 policy")
            policy = mapping[policy_key]

        valid_policies = {
            ELECTRON_GFP2_POLICY_PURE,
            ELECTRON_GFP2_POLICY_REGIME,
        }
        if policy not in valid_policies:
            print_error("Unknown electron GFP2 policy")

        self.electron_gfp2_policy = policy

    def set_electron_gfp2_scheme(self, scheme):
        if isinstance(scheme, str):
            scheme_key = scheme.strip().lower()
            mapping = {
                "kernel": ELECTRON_GFP2_SCHEME_KERNEL,
                "elastic_only": ELECTRON_GFP2_SCHEME_ELASTIC_ONLY,
            }
            if scheme_key not in mapping:
                print_error("Unknown electron GFP2 scheme")
            scheme = mapping[scheme_key]

        valid_schemes = {
            ELECTRON_GFP2_SCHEME_KERNEL,
            ELECTRON_GFP2_SCHEME_ELASTIC_ONLY,
        }
        if scheme not in valid_schemes:
            print_error("Unknown electron GFP2 scheme")

        self.electron_gfp2_scheme = scheme
