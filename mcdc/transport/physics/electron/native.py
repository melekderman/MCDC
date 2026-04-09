import math
import numpy as np

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.numba_types as type_
import mcdc.transport.particle as particle_module
import mcdc.transport.particle_bank as particle_bank_module
import mcdc.transport.rng as rng

from mcdc.constant import (
    ELECTRON_CUTOFF_ENERGY,
    ELECTRON_GFP2_POLICY_PURE,
    ELECTRON_GFP2_SCHEME_KERNEL,
    ELECTRON_GFP2_STATE_0,
    ELECTRON_GFP2_STATE_1,
    ELECTRON_MASS,
    ELECTRON_REACTION_BREMSSTRAHLUNG,
    ELECTRON_REACTION_EXCITATION,
    ELECTRON_REACTION_ELASTIC_SCATTERING,
    ELECTRON_REACTION_IONIZATION,
    ELECTRON_REACTION_TOTAL,
    ELECTRON_ELASTIC_MODE_COUPLED,
    ELECTRON_ELASTIC_MODE_DECOUPLED,
    ELECTRON_ELASTIC_MODE_GFP2,
    INF,
    LIGHT_SPEED,
    PI,
)
from mcdc.transport.data import evaluate_data
from mcdc.transport.distribution import (
    sample_distribution,
    sample_multi_table_cdf,
)
from mcdc.transport.physics.util import (
    evaluate_electron_xs_energy_grid,
    scatter_direction,
)
from mcdc.transport.util import linear_interpolation

# ======================================================================================
# Particle attributes
# ======================================================================================


@njit
def particle_speed(particle_container):
    particle = particle_container[0]
    E = particle["E"]
    mass = ELECTRON_MASS
    return LIGHT_SPEED * math.sqrt(E * (E + 2.0 * mass)) / (E + mass)


@njit
def collision_distance(particle_container, mcdc, data):
    particle = particle_container[0]
    elastic_mode = mcdc["settings"]["electron_elastic_mode"]

    if (
        elastic_mode == ELECTRON_ELASTIC_MODE_GFP2
        and particle["g"] == ELECTRON_GFP2_STATE_1
    ):
        return 0.0

    SigmaT = macro_xs(ELECTRON_REACTION_TOTAL, particle_container, mcdc, data)
    if SigmaT == 0.0:
        return INF

    xi = rng.lcg(particle_container)
    return -math.log(xi) / SigmaT


# ======================================================================================
# Material properties
# ======================================================================================


@njit
def macro_xs(reaction_type, particle_container, mcdc, data):
    particle = particle_container[0]
    material = mcdc["native_materials"][particle["material_ID"]]
    E = particle["E"]
    elastic_mode = mcdc["settings"]["electron_elastic_mode"]
    gfp2_scheme = mcdc["settings"]["electron_gfp2_scheme"]
    gfp2_state = particle["g"]

    total = 0.0
    for i in range(material["N_element"]):
        element_ID = int(mcdc_get.native_material.element_IDs(i, material, data))
        element = mcdc["elements"][element_ID]

        element_density = mcdc_get.native_material.element_densities(i, material, data)
        if elastic_mode == ELECTRON_ELASTIC_MODE_GFP2:
            if (
                gfp2_state == ELECTRON_GFP2_STATE_1
                or gfp2_scheme == ELECTRON_GFP2_SCHEME_KERNEL
            ):
                xs = total_micro_xs_gfp2(
                    reaction_type, E, element, gfp2_state, mcdc, data
                )
            else:
                xs = total_micro_xs(reaction_type, E, element, data)
        else:
            xs = total_micro_xs(reaction_type, E, element, data)
        total += element_density * xs

    return total


@njit
def total_micro_xs(reaction_type, E, element, data):
    idx, E0, E1 = evaluate_electron_xs_energy_grid(E, element, data)
    if reaction_type == ELECTRON_REACTION_TOTAL:
        xs0 = mcdc_get.element.electron_total_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_total_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_IONIZATION:
        xs0 = mcdc_get.element.electron_ionization_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_ionization_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_ELASTIC_SCATTERING:
        xs0 = mcdc_get.element.electron_elastic_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_elastic_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_EXCITATION:
        xs0 = mcdc_get.element.electron_excitation_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_excitation_xs(idx + 1, element, data)
    elif reaction_type == ELECTRON_REACTION_BREMSSTRAHLUNG:
        xs0 = mcdc_get.element.electron_bremsstrahlung_xs(idx, element, data)
        xs1 = mcdc_get.element.electron_bremsstrahlung_xs(idx + 1, element, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def reaction_micro_xs(E, reaction_base, element, data):
    idx, E0, E1 = evaluate_electron_xs_energy_grid(E, element, data)

    # Apply offset
    offset = reaction_base["xs_offset_"]
    if idx < offset:
        return 0.0
    else:
        idx -= offset

    xs0 = mcdc_get.electron_reaction.xs(idx, reaction_base, data)
    xs1 = mcdc_get.electron_reaction.xs(idx + 1, reaction_base, data)
    return linear_interpolation(E, E0, E1, xs0, xs1)


@njit
def element_nonelastic_total_xs(E, element, data):
    return (
        total_micro_xs(ELECTRON_REACTION_IONIZATION, E, element, data)
        + total_micro_xs(ELECTRON_REACTION_BREMSSTRAHLUNG, E, element, data)
        + total_micro_xs(ELECTRON_REACTION_EXCITATION, E, element, data)
    )


@njit
def gfp2_transition_rate_xs(E, reaction, mcdc, data):
    data_base = mcdc["data"][int(reaction["gfp2_transition_rate_ID"])]
    return evaluate_data(E, data_base, mcdc, data)


@njit
def gfp2_sigma_delta0_xs(E, reaction, mcdc, data):
    data_base = mcdc["data"][int(reaction["gfp2_sigma_delta0_ID"])]
    return evaluate_data(E, data_base, mcdc, data)


@njit
def reaction_gfp2_enabled(E, reaction, mcdc, data):
    policy = mcdc["settings"]["electron_gfp2_policy"]
    if policy == ELECTRON_GFP2_POLICY_PURE:
        return True

    transition_rate = gfp2_transition_rate_xs(E, reaction, mcdc, data)
    sigma_delta0 = gfp2_sigma_delta0_xs(E, reaction, mcdc, data)
    return transition_rate > 0.0 and sigma_delta0 > 0.0


@njit
def element_elastic_state0_total_xs_gfp2(E, element, mcdc, data):
    total = 0.0
    for i in range(element["N_electron_elastic_scattering_reaction"]):
        reaction_ID = int(
            mcdc_get.element.electron_elastic_scattering_reaction_IDs(i, element, data)
        )
        reaction = mcdc["electron_elastic_scattering_reactions"][reaction_ID]

        if reaction_gfp2_enabled(E, reaction, mcdc, data):
            total += gfp2_transition_rate_xs(E, reaction, mcdc, data)
        else:
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = mcdc["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

    return total


@njit
def element_gfp2_transition_total_xs(E, element, mcdc, data):
    total = 0.0
    for i in range(element["N_electron_elastic_scattering_reaction"]):
        reaction_ID = int(
            mcdc_get.element.electron_elastic_scattering_reaction_IDs(i, element, data)
        )
        reaction = mcdc["electron_elastic_scattering_reactions"][reaction_ID]
        if reaction_gfp2_enabled(E, reaction, mcdc, data):
            total += gfp2_transition_rate_xs(E, reaction, mcdc, data)
    return total


@njit
def element_gfp2_state1_total_xs(E, element, mcdc, data):
    total = 0.0
    for i in range(element["N_electron_elastic_scattering_reaction"]):
        reaction_ID = int(
            mcdc_get.element.electron_elastic_scattering_reaction_IDs(i, element, data)
        )
        reaction = mcdc["electron_elastic_scattering_reactions"][reaction_ID]
        if not reaction_gfp2_enabled(E, reaction, mcdc, data):
            continue
        total += gfp2_sigma_delta0_xs(E, reaction, mcdc, data)
        total += gfp2_transition_rate_xs(E, reaction, mcdc, data)
    return total


@njit
def total_micro_xs_gfp2(reaction_type, E, element, gfp2_state, mcdc, data):
    if reaction_type == ELECTRON_REACTION_TOTAL:
        if gfp2_state == ELECTRON_GFP2_STATE_1:
            return element_gfp2_state1_total_xs(E, element, mcdc, data)
        return element_nonelastic_total_xs(
            E, element, data
        ) + element_elastic_state0_total_xs_gfp2(E, element, mcdc, data)

    if reaction_type == ELECTRON_REACTION_ELASTIC_SCATTERING:
        if gfp2_state == ELECTRON_GFP2_STATE_1:
            return element_gfp2_state1_total_xs(E, element, mcdc, data)
        return element_elastic_state0_total_xs_gfp2(E, element, mcdc, data)

    return total_micro_xs(reaction_type, E, element, data)


# ======================================================================================
# Collision
# ======================================================================================


@njit
def collision(particle_container, collision_data_container, mcdc, data):
    particle = particle_container[0]
    collision_data = collision_data_container[0]
    material = mcdc["native_materials"][particle["material_ID"]]
    elastic_mode = mcdc["settings"]["electron_elastic_mode"]
    gfp2_scheme = mcdc["settings"]["electron_gfp2_scheme"]

    # Particle properties
    E = particle["E"]

    # Check for cutoff energy
    if E <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    if (
        elastic_mode == ELECTRON_ELASTIC_MODE_GFP2
        and particle["g"] == ELECTRON_GFP2_STATE_1
    ):
        gfp2_state1_collision(
            particle_container, collision_data_container, material, mcdc, data
        )
        return

    # ==================================================================================
    # Sample colliding element
    # ==================================================================================

    SigmaT = macro_xs(ELECTRON_REACTION_TOTAL, particle_container, mcdc, data)

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    for i in range(material["N_element"]):
        element_ID = int(mcdc_get.native_material.element_IDs(i, material, data))
        element = mcdc["elements"][element_ID]

        element_density = mcdc_get.native_material.element_densities(i, material, data)
        if (
            elastic_mode == ELECTRON_ELASTIC_MODE_GFP2
            and gfp2_scheme == ELECTRON_GFP2_SCHEME_KERNEL
        ):
            sigmaT = total_micro_xs_gfp2(
                ELECTRON_REACTION_TOTAL,
                E,
                element,
                ELECTRON_GFP2_STATE_0,
                mcdc,
                data,
            )
        else:
            sigmaT = total_micro_xs(ELECTRON_REACTION_TOTAL, E, element, data)

        total += element_density * sigmaT

        if total > xi:
            break

    # ==================================================================================
    # Sample and perform reaction
    # ==================================================================================

    sigma_ionization = total_micro_xs(ELECTRON_REACTION_IONIZATION, E, element, data)
    if (
        elastic_mode == ELECTRON_ELASTIC_MODE_GFP2
        and gfp2_scheme == ELECTRON_GFP2_SCHEME_KERNEL
    ):
        sigma_elastic = element_elastic_state0_total_xs_gfp2(E, element, mcdc, data)
    else:
        sigma_elastic = total_micro_xs(
            ELECTRON_REACTION_ELASTIC_SCATTERING, E, element, data
        )
    sigma_bremsstrahlung = total_micro_xs(
        ELECTRON_REACTION_BREMSSTRAHLUNG, E, element, data
    )
    sigma_excitation = total_micro_xs(ELECTRON_REACTION_EXCITATION, E, element, data)

    xi = rng.lcg(particle_container) * sigmaT
    total = 0.0

    # Ionization
    total += sigma_ionization
    if xi < total:
        total -= sigma_ionization
        for i in range(element["N_electron_ionization_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_ionization_reaction_IDs(i, element, data)
            )
            reaction = mcdc["electron_ionization_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = mcdc["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                ionization(
                    reaction,
                    sigma_ionization,
                    particle_container,
                    collision_data_container,
                    element,
                    mcdc,
                    data,
                )
                return

    # Elastic scattering
    total += sigma_elastic
    if xi < total:
        total -= sigma_elastic
        for i in range(element["N_electron_elastic_scattering_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_elastic_scattering_reaction_IDs(
                    i, element, data
                )
            )
            reaction = mcdc["electron_elastic_scattering_reactions"][reaction_ID]
            use_gfp2 = (
                elastic_mode == ELECTRON_ELASTIC_MODE_GFP2
                and reaction_gfp2_enabled(E, reaction, mcdc, data)
            )

            if (
                elastic_mode == ELECTRON_ELASTIC_MODE_GFP2
                and gfp2_scheme == ELECTRON_GFP2_SCHEME_KERNEL
            ):
                if use_gfp2:
                    total += gfp2_transition_rate_xs(E, reaction, mcdc, data)
                else:
                    reaction_base_ID = reaction["parent_ID"]
                    reaction_base = mcdc["electron_reactions"][reaction_base_ID]
                    total += reaction_micro_xs(E, reaction_base, element, data)
            else:
                reaction_base_ID = reaction["parent_ID"]
                reaction_base = mcdc["electron_reactions"][reaction_base_ID]
                total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                if use_gfp2:
                    particle["g"] = ELECTRON_GFP2_STATE_1
                    gfp2_state1_relaxation(
                        particle_container,
                        collision_data_container,
                        material,
                        mcdc,
                        data,
                    )
                else:
                    elastic_scattering(
                        reaction, particle_container, element, mcdc, data
                    )
                return

    # Bremsstrahlung
    total += sigma_bremsstrahlung
    if xi < total:
        total -= sigma_bremsstrahlung
        for i in range(element["N_electron_bremsstrahlung_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_bremsstrahlung_reaction_IDs(i, element, data)
            )
            reaction = mcdc["electron_bremsstrahlung_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = mcdc["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                bremsstrahlung(
                    reaction, particle_container, collision_data_container, mcdc, data
                )
                return

    # Excitation
    total += sigma_excitation
    if xi < total:
        total -= sigma_excitation
        for i in range(element["N_electron_excitation_reaction"]):
            reaction_ID = int(
                mcdc_get.element.electron_excitation_reaction_IDs(i, element, data)
            )
            reaction = mcdc["electron_excitation_reactions"][reaction_ID]
            reaction_base_ID = reaction["parent_ID"]
            reaction_base = mcdc["electron_reactions"][reaction_base_ID]
            total += reaction_micro_xs(E, reaction_base, element, data)

            if xi < total:
                excitation(
                    reaction, particle_container, collision_data_container, mcdc, data
                )
                return


# ======================================================================================
# Elastic scattering
# ======================================================================================


@njit
def elastic_scattering(reaction, particle_container, element, mcdc, data):
    particle = particle_container[0]

    # Current energy
    E = particle["E"]
    Z = int(element["atomic_number"])
    mu_cut = float(reaction["mu_cut"])

    # -------------------------------------------------------------------------
    # Total elastic xs
    # -------------------------------------------------------------------------

    reaction_base_ID = int(reaction["parent_ID"])
    reaction_base = mcdc["electron_reactions"][reaction_base_ID]
    xs_total = reaction_micro_xs(E, reaction_base, element, data)

    # If large-angle, xs from data table
    xs_large = elastic_large_xs(E, reaction, mcdc, data)

    # -------------------------------------------------------------------------
    # Sample elastic scattering cosine according to the selected method
    # -------------------------------------------------------------------------
    elastic_mode = mcdc["settings"]["electron_elastic_mode"]
    if elastic_mode == ELECTRON_ELASTIC_MODE_DECOUPLED:
        xi = rng.lcg(particle_container) * xs_total
        if xi < xs_large:
            # Large-angle: sample from EEDL table
            multi_table = mcdc["multi_table_distributions"][reaction["mu_ID"]]
            mu0 = sample_multi_table_cdf(E, particle_container, multi_table, data)
        else:
            # Small-angle: analytical SR CDF inversion over [mu_cut, 1]
            mu0 = sample_small_angle_mu_coulomb(E, Z, particle_container, mu_cut)
    elif elastic_mode == ELECTRON_ELASTIC_MODE_COUPLED:
        multi_table = mcdc["multi_table_distributions"][reaction["mu_coupled_ID"]]
        mu0 = sample_multi_table_cdf(E, particle_container, multi_table, data)
    elif elastic_mode == ELECTRON_ELASTIC_MODE_GFP2:
        xi = rng.lcg(particle_container) * xs_total
        if xi < xs_large:
            multi_table = mcdc["multi_table_distributions"][reaction["mu_ID"]]
            mu0 = sample_multi_table_cdf(E, particle_container, multi_table, data)
        else:
            mu0 = sample_small_angle_mu_coulomb(E, Z, particle_container, mu_cut)
    else:
        xi = rng.lcg(particle_container) * xs_total
        if xi < xs_large:
            multi_table = mcdc["multi_table_distributions"][reaction["mu_ID"]]
            mu0 = sample_multi_table_cdf(E, particle_container, multi_table, data)
        else:
            mu0 = sample_small_angle_mu_coulomb(E, Z, particle_container, mu_cut)

    # Update direction
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    ux_new, uy_new, uz_new = scatter_direction(ux, uy, uz, mu0, azi)

    particle["ux"] = ux_new
    particle["uy"] = uy_new
    particle["uz"] = uz_new


@njit
def compute_scattering_eta(E, Z):
    pc = math.sqrt(E * (E + 2.0 * ELECTRON_MASS))
    beta = pc / (E + ELECTRON_MASS)
    tau = E / ELECTRON_MASS
    FINE_STRUCTURE_CONSTANT = 7.2973525693e-3

    r = (FINE_STRUCTURE_CONSTANT * ELECTRON_MASS) / (0.885 * pc)
    z_sq = float(Z) ** (2.0 / 3.0)
    bracket = 1.13 + 3.76 * ((FINE_STRUCTURE_CONSTANT * float(Z)) / beta) ** 2
    rel = math.sqrt(tau / (tau + 1.0))

    return 0.25 * (r * r) * z_sq * bracket * rel


@njit
def sr_cdf(mu, eta):
    """
    Screened Rutherford CDF: fraction of events with scattering cosine < mu.

    F(mu) = [1/eta - 1/(1-mu+2*eta)] / [1/eta - 1/(2+2*eta)]
    """
    num = (1.0 / eta) - 1.0 / (1.0 - mu + 2.0 * eta)
    denom = (1.0 / eta) - 1.0 / (2.0 + 2.0 * eta)
    if denom == 0.0:
        return 1.0
    result = num / denom
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result


@njit
def sample_small_angle_mu_coulomb(E, Z, rng_state, mu_cut):
    eta = compute_scattering_eta(E, Z)

    x_cut = 1.0 - mu_cut
    u = rng.lcg(rng_state)

    denom = (1.0 / eta) - (1.0 / (eta + x_cut))
    inv = (1.0 / eta) - u * denom
    x = (1.0 / inv) - eta

    return 1.0 - x


@njit
def elastic_large_xs(E, reaction, mcdc, data):
    data_base = mcdc["data"][int(reaction["xs_large_ID"])]
    return evaluate_data(E, data_base, mcdc, data)


@njit
def gfp2_state1_collision(
    particle_container, collision_data_container, material, mcdc, data
):
    particle = particle_container[0]
    E = particle["E"]

    SigmaT = macro_xs(ELECTRON_REACTION_TOTAL, particle_container, mcdc, data)
    if SigmaT <= 0.0:
        particle["g"] = ELECTRON_GFP2_STATE_0
        return

    xi = rng.lcg(particle_container) * SigmaT
    total = 0.0
    sigmaT = 0.0
    element = mcdc["elements"][0]
    for i in range(material["N_element"]):
        element_ID = int(mcdc_get.native_material.element_IDs(i, material, data))
        element = mcdc["elements"][element_ID]
        element_density = mcdc_get.native_material.element_densities(i, material, data)
        sigmaT = element_gfp2_state1_total_xs(E, element, mcdc, data)
        total += element_density * sigmaT
        if total > xi:
            break

    if sigmaT <= 0.0:
        particle["g"] = ELECTRON_GFP2_STATE_0
        return

    xi = rng.lcg(particle_container) * sigmaT
    total = 0.0
    for i in range(element["N_electron_elastic_scattering_reaction"]):
        reaction_ID = int(
            mcdc_get.element.electron_elastic_scattering_reaction_IDs(i, element, data)
        )
        reaction = mcdc["electron_elastic_scattering_reactions"][reaction_ID]

        if not reaction_gfp2_enabled(E, reaction, mcdc, data):
            continue

        total += gfp2_sigma_delta0_xs(E, reaction, mcdc, data)
        if xi < total:
            gfp2_state1_scatter(reaction, particle_container)
            return

        total += gfp2_transition_rate_xs(E, reaction, mcdc, data)
        if xi < total:
            particle["g"] = ELECTRON_GFP2_STATE_0
            return

    particle["g"] = ELECTRON_GFP2_STATE_0


@njit
def gfp2_state1_relaxation(
    particle_container, collision_data_container, material, mcdc, data
):
    particle = particle_container[0]

    # State-1 has no streaming or slowing-down, so we can resolve its local
    # scatter/transition chain before returning control to the transport loop.
    while particle["alive"] and particle["g"] == ELECTRON_GFP2_STATE_1:
        gfp2_state1_collision(
            particle_container, collision_data_container, material, mcdc, data
        )


@njit
def gfp2_state1_scatter(reaction, particle_container):
    particle = particle_container[0]
    mu0 = float(reaction["gfp2_mu_star"])
    azi = 2.0 * PI * rng.lcg(particle_container)
    ux_new, uy_new, uz_new = scatter_direction(
        particle["ux"], particle["uy"], particle["uz"], mu0, azi
    )
    particle["ux"] = ux_new
    particle["uy"] = uy_new
    particle["uz"] = uz_new
    particle["g"] = ELECTRON_GFP2_STATE_1


# ======================================================================================
# Excitation
# ======================================================================================


@njit
def excitation(reaction, particle_container, collision_data_container, mcdc, data):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    # Current energy
    E = particle["E"]

    dE = evaluate_eloss(E, reaction, mcdc, data)

    # Calculate outgoing energy
    E_out = E - dE

    # Check for cutoff
    if E_out <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["E"] = 0.0
        particle["alive"] = False
        return

    # If above cutoff, just deposit dE
    particle["E"] = E_out
    collision_data["energy_deposition"] += dE * particle["w"]


@njit
def evaluate_eloss(E, reaction, mcdc, data):
    data_base = mcdc["data"][int(reaction["eloss_ID"])]
    return evaluate_data(E, data_base, mcdc, data)


# ======================================================================================
# Bremsstrahlung
# ======================================================================================


@njit
def bremsstrahlung(reaction, particle_container, collision_data_container, mcdc, data):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    # Current energy
    E = particle["E"]

    dE = evaluate_eloss(E, reaction, mcdc, data)
    E_out = E - dE

    # Check for cutoff
    if E_out <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E_out * particle["w"]
        particle["E"] = 0.0
        particle["alive"] = False
        return

    # If above cutoff, allow photon to escape and update electron energy
    particle["E"] = E_out


# ======================================================================================
# Ionization
# ======================================================================================


@njit
def ionization(
    reaction,
    sigma_ionization,
    particle_container,
    collision_data_container,
    element,
    mcdc,
    data,
):
    particle = particle_container[0]
    collision_data = collision_data_container[0]

    # Current energy
    E = particle["E"]

    # Sample subshell
    N = int(reaction["N_subshell"])
    xs_vals = np.empty(N, dtype=np.float64)

    for i in range(N):
        xs_sub_ID = int(
            mcdc_get.electron_ionization_reaction.subshell_x_IDs(i, reaction, data)
        )
        xs_sub_table = mcdc["data"][xs_sub_ID]
        xs_sub_i = evaluate_data(E, xs_sub_table, mcdc, data)
        xs_vals[i] = xs_sub_i

    xi = rng.lcg(particle_container) * sigma_ionization
    total_acc = 0.0
    # If shell sums miss the total slightly, fall back to the last shell.
    chosen = N - 1
    for i in range(N):
        total_acc += xs_vals[i]
        if total_acc >= xi:
            chosen = i
            break

    # Binding energy
    B = mcdc_get.element.electron_ionization_subshell_binding_energy(
        chosen, element, data
    )
    if E <= B:
        collision_data["energy_deposition"] += E * particle["w"]
        particle["alive"] = False
        particle["E"] = 0.0
        return

    # Sample secondary energy
    dist_ID = int(
        mcdc_get.electron_ionization_reaction.subshell_product_IDs(
            chosen, reaction, data
        )
    )
    dist_base = mcdc["distributions"][dist_ID]
    T_delta = sample_distribution(
        E, dist_base, particle_container, mcdc, data, scale=False
    )

    # Primary outgoing energy
    E_out = E - B - T_delta
    particle["E"] = E_out

    collision_data["energy_deposition"] += B * particle["w"]

    primary_alive_after = True
    if E_out <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += E_out * particle["w"]
        particle["E"] = 0.0
        particle["alive"] = False
        primary_alive_after = False

    if T_delta <= ELECTRON_CUTOFF_ENERGY:
        collision_data["energy_deposition"] += T_delta * particle["w"]
        return

    # Sample delta direction
    ux_delta, uy_delta, uz_delta = sample_delta_direction(
        T_delta, E, particle_container
    )

    # Momentum conservation if primary survives
    if primary_alive_after:
        p_before = math.sqrt(E * (E + 2.0 * ELECTRON_MASS))
        p_delta = math.sqrt(T_delta * (T_delta + 2.0 * ELECTRON_MASS))

        ux_before = particle["ux"]
        uy_before = particle["uy"]
        uz_before = particle["uz"]

        # Momentum vectors after collision
        px_after = p_before * ux_before - p_delta * ux_delta
        py_after = p_before * uy_before - p_delta * uy_delta
        pz_after = p_before * uz_before - p_delta * uz_delta

        # Normalize and set primary's new direction
        norm_sq = px_after * px_after + py_after * py_after + pz_after * pz_after
        if norm_sq > 0.0:
            norm = math.sqrt(norm_sq)
            particle["ux"] = px_after / norm
            particle["uy"] = py_after / norm
            particle["uz"] = pz_after / norm

    # Add secondary particle to bank
    particle_container_new = np.zeros(1, type_.particle_data)
    particle_new = particle_container_new[0]
    particle_module.copy_as_child(particle_container_new, particle_container)

    particle_new["E"] = T_delta
    particle_new["ux"] = ux_delta
    particle_new["uy"] = uy_delta
    particle_new["uz"] = uz_delta
    particle_new["g"] = ELECTRON_GFP2_STATE_0
    particle_new["w"] = particle["w"]

    particle_bank_module.bank_active_particle(particle_container_new, mcdc)


@njit
def compute_mu_delta(T_delta, T_prim):
    pd = math.sqrt(T_delta * (T_delta + 2.0 * ELECTRON_MASS))
    pp = math.sqrt(T_prim * (T_prim + 2.0 * ELECTRON_MASS))
    mu = (T_delta * (T_prim + 2.0 * ELECTRON_MASS)) / (pd * pp)

    # Check in case of numerical issues
    if mu < -1.0:
        mu = -1.0
    if mu > 1.0:
        mu = 1.0

    return mu


@njit
def sample_delta_direction(T_delta, T_prim, particle_container):
    particle = particle_container[0]
    mu = compute_mu_delta(T_delta, T_prim)
    azi = 2.0 * PI * rng.lcg(particle_container)
    return scatter_direction(particle["ux"], particle["uy"], particle["uz"], mu, azi)
