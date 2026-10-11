from math import isfinite
from numbers import Real

from numpy import float64
from numpy.typing import NDArray

####

import mcdc.object_.distribution as distribution

from mcdc.constant import (
    ELECTRON_REACTION_BREMSSTRAHLUNG,
    ELECTRON_REACTION_EXCITATION,
    ELECTRON_REACTION_ELASTIC_SCATTERING,
    ELECTRON_REACTION_IONIZATION,
    INTERPOLATION_LINEAR,
    MU_CUTOFF,
    REFERENCE_FRAME_COM,
    REFERENCE_FRAME_LAB,
)
from mcdc.object_.secondary_product import SecondaryProduct
from mcdc.object_.base import MCDCPolymorphic
from mcdc.object_.data import DataBase, DataTable
from mcdc.object_.distribution import DistributionBase, DistributionMultiTable
from mcdc.print_ import print_1d_array, print_error

# ======================================================================================
# Electron reaction base class
# ======================================================================================


def read_energy(dataset):
    """Read an electron energy dataset in MC/DC's internal eV units."""

    values = dataset[()]
    unit = dataset.attrs.get("unit", "eV")
    if unit == "MeV":
        values = values * 1e6
    return values


class ElectronReactionBase(MCDCPolymorphic):
    """Base electron reaction data loaded from the HDF5 physics library.

    Stores the ENDF reaction identifier, cross-section segment and its offset
    into the element energy grid, and the reaction reference frame.
    """

    # MC/DC framework metadata
    label = "electron_reaction"
    sub_type = -1  # Polymorphic base

    MT: int
    xs: NDArray[float64]
    xs_offset_: int  # "xs_offset" is reserved for "xs"
    reference_frame: int

    secondary_products: list[SecondaryProduct]

    def __init__(self, MT, xs, xs_offset, reference_frame):
        super().__init__()
        self.secondary_products = []
        self.MT = MT
        self.xs = xs
        self.xs_offset_ = xs_offset
        self.reference_frame = reference_frame

    def __repr__(self):
        text = super().__repr__()

        text += f"  - ID: {self.ID}\n"
        text += f"  - MT: {self.MT}\n"
        text += f"  - XS {print_1d_array(self.xs)} barn\n"
        text += f"  - Reference frame: {decode_reference_frame(self.reference_frame)}\n"
        return text


def decode_reference_frame(type_):
    """Return the display name for a packed reference-frame code."""

    if type_ == REFERENCE_FRAME_LAB:
        return "Laboratory"
    elif type_ == REFERENCE_FRAME_COM:
        return "Center of mass"


# ======================================================================================
# Electron ionization
# ======================================================================================


class ElectronReactionIonization(ElectronReactionBase):
    """Electron ionization reaction with subshell cross sections and products."""

    # MC/DC framework metadata
    label = "electron_ionization_reaction"
    sub_type = ELECTRON_REACTION_IONIZATION

    N_subshell: int
    subshell_xs: list[DataBase]
    subshell_product: list[DistributionBase]

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        subshell_xs,
        subshell_product,
    ):
        super().__init__(MT, xs, xs_offset, reference_frame)

        self.N_subshell = len(subshell_xs)
        self.subshell_xs = subshell_xs
        self.subshell_product = subshell_product

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an ionization reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        subshells = h5_group["subshells"]
        subshell_names = list(subshells.keys())

        subshell_xs = []
        subshell_product = []

        for name in subshell_names:
            subshell = subshells[name]

            # Subshell cross section table (each has its own energy grid)
            subshell_xs.append(
                DataTable(
                    read_energy(subshell["energy_grid"]),
                    subshell["xs"][()],
                    INTERPOLATION_LINEAR,
                )
            )

            # Secondary electron energy distribution. The sampler skips the
            # tables at or below the subshell binding energy.
            product = subshell["product"]
            product_grid = read_energy(product["energy_grid"])
            binding_energy = float(read_energy(subshell["binding_energy"]))
            if product_grid[-1] <= binding_energy:
                print_error(
                    f"Ionization subshell {name} has no product spectrum above "
                    "its binding energy."
                )
            if "CDF" in product:
                subshell_product.append(
                    DistributionMultiTable(
                        product_grid,
                        product["energy_offset"][()],
                        read_energy(product["value"]),
                        cdf=product["CDF"][()],
                    )
                )
            else:
                subshell_product.append(
                    DistributionMultiTable(
                        product_grid,
                        product["energy_offset"][()],
                        read_energy(product["value"]),
                        product["PDF"][()],
                    )
                )

        return cls(
            MT,
            xs,
            xs_offset,
            reference_frame,
            subshell_xs,
            subshell_product,
        )

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Number of subshells: {self.N_subshell}\n"
        for i in range(self.N_subshell):
            text += f"    - Subshell {i+1}\n"
            text += f"      - XS: DataTable [ID: {self.subshell_xs[i].ID}]\n"
            product = self.subshell_product[i]
            text += f"      - Secondary electron spectrum: {distribution.decode_type(product.type)} [ID: {product.ID}]\n"
        return text


# ======================================================================================
# Electron elastic scattering
# ======================================================================================


class ElectronReactionElasticScattering(ElectronReactionBase):
    """Elastic electron scattering with tabulated and analytic components.

    ``mu_cut`` is the analytic component's lower cosine bound in [-1, 1),
    defaulting to ``MU_CUTOFF`` (0.999999).
    It does not change the tabulated component's support or cross section.
    """

    # MC/DC framework metadata
    label = "electron_elastic_scattering_reaction"
    sub_type = ELECTRON_REACTION_ELASTIC_SCATTERING

    mu_cut: float
    xs_large: DataBase
    mu: DistributionMultiTable

    def __init__(
        self,
        MT,
        xs,
        xs_offset,
        reference_frame,
        xs_large,
        mu,
        *,
        mu_cut=MU_CUTOFF,
    ):
        super().__init__(MT, xs, xs_offset, reference_frame)
        if (
            not isinstance(mu_cut, Real)
            or not isfinite(mu_cut)
            or not -1.0 <= mu_cut < 1.0
        ):
            print_error("Elastic mu_cut must be finite and satisfy -1 <= mu_cut < 1.")
        self.mu_cut = float(mu_cut)
        if not isfinite(self.mu_cut) or not -1.0 <= self.mu_cut < 1.0:
            print_error("Elastic mu_cut must be finite and satisfy -1 <= mu_cut < 1.")
        self.xs_large = xs_large
        self.mu = mu

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an elastic-scattering reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        large_angle = h5_group["large_angle"]
        xs_large = DataTable(
            read_energy(large_angle["xs_energy"]),
            large_angle["xs"][()],
            INTERPOLATION_LINEAR,
        )

        mu_group = large_angle["scattering_cosine"]
        if "CDF" in mu_group:
            mu = DistributionMultiTable(
                read_energy(mu_group["energy_grid"]),
                mu_group["energy_offset"][()],
                mu_group["value"][()],
                cdf=mu_group["CDF"][()],
            )
        else:
            mu = DistributionMultiTable(
                read_energy(mu_group["energy_grid"]),
                mu_group["energy_offset"][()],
                mu_group["value"][()],
                mu_group["PDF"][()],
            )

        # PyEPICS stores small_angle beside the MT reaction group.
        small_angle = h5_group.parent.get("small_angle")
        mu_cut = (
            small_angle.attrs.get("mu_cut", MU_CUTOFF)
            if small_angle is not None
            else MU_CUTOFF
        )

        return cls(MT, xs, xs_offset, reference_frame, xs_large, mu, mu_cut=mu_cut)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Mu cut: {self.mu_cut}\n"
        text += f"  - Large angle XS: DataTable [ID: {self.xs_large.ID}]\n"
        text += f"  - Scattering cosine: {distribution.decode_type(self.mu.type)} [ID: {self.mu.ID}]\n"
        return text


# ======================================================================================
# Electron bremsstrahlung
# ======================================================================================


class ElectronReactionBremsstrahlung(ElectronReactionBase):
    """Electron bremsstrahlung reaction with tabulated energy loss."""

    # MC/DC framework metadata
    label = "electron_bremsstrahlung_reaction"
    sub_type = ELECTRON_REACTION_BREMSSTRAHLUNG

    eloss: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.eloss = eloss

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build a bremsstrahlung reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        base = h5_group["energy_loss"]
        eloss = DataTable(
            read_energy(base["energy"]),
            read_energy(base["value"]),
            INTERPOLATION_LINEAR,
        )

        return cls(MT, xs, xs_offset, reference_frame, eloss)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: DataTable [ID: {self.eloss.ID}]\n"
        return text


# ======================================================================================
# Electron excitation
# ======================================================================================


class ElectronReactionExcitation(ElectronReactionBase):
    """Electron excitation reaction with tabulated energy loss."""

    # MC/DC framework metadata
    label = "electron_excitation_reaction"
    sub_type = ELECTRON_REACTION_EXCITATION

    eloss: DataBase

    def __init__(self, MT, xs, xs_offset, reference_frame, eloss):
        super().__init__(MT, xs, xs_offset, reference_frame)
        self.eloss = eloss

    @classmethod
    def from_h5_group(cls, h5_group):
        """Build an excitation reaction from a library HDF5 group."""
        MT, xs, xs_offset, reference_frame = set_basic_properties(h5_group)

        base = h5_group["energy_loss"]
        eloss = DataTable(
            read_energy(base["energy"]),
            read_energy(base["value"]),
            INTERPOLATION_LINEAR,
        )

        return cls(MT, xs, xs_offset, reference_frame, eloss)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Energy loss: DataTable [ID: {self.eloss.ID}]\n"
        return text


# ======================================================================================
# Helper functions
# ======================================================================================


def set_basic_properties(h5_group):
    """Read properties shared by all electron reactions from an HDF5 group."""

    MT = h5_group.attrs["MT"][()]
    xs = h5_group["xs"][()]
    xs_offset = h5_group["xs"].attrs["offset"]
    reference_frame = h5_group["reference_frame"][()].decode("utf-8")
    if reference_frame == "LAB":
        reference_frame = REFERENCE_FRAME_LAB
    elif reference_frame == "COM":
        reference_frame = REFERENCE_FRAME_COM
    return MT, xs, xs_offset, reference_frame
