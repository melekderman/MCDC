import importlib
import os
import sys

import numpy as np

# ======================================================================================
# CGMF interface
# ======================================================================================
#
# Optional dependency: cgmfwrap (https://github.com/melekderman/cgmfwrap) - TEMPORARYLY.
# TODO: Remove this module and the cgmfwrap dependency once CGMF support is implemented
# directly in C++ and exposed through pybind11, so that CGMF sampling can be used from 
# the Numba transport kernel without `objmode` and without the overhead of a Python callback.
# Set CGMFWRAP_PATH to the directory containing the built cgmfwrap module,
# unless it is already importable from the active Python environment.

_CGMFWRAP = None
_MAX_EVENT_RETRIES = 32


def _load_cgmfwrap():
    global _CGMFWRAP

    if _CGMFWRAP is not None:
        return _CGMFWRAP

    path = os.environ.get("CGMFWRAP_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)

    try:
        _CGMFWRAP = importlib.import_module("cgmfwrap")
    except ImportError as exc:
        raise RuntimeError(
            "CGMF emission requested, but cgmfwrap could not be imported. "
            "Build cgmfwrap (https://github.com/melekderman/cgmfwrap) and "
            "either install it into the active environment or set "
            "CGMFWRAP_PATH to its build directory."
        ) from exc

    return _CGMFWRAP


# ======================================================================================
# Event sampling
# ======================================================================================


def run_event(zaid, incident_energy_eV):
    """Sample one CGMF fission event.

    Retries up to _MAX_EVENT_RETRIES times if CGMF returns an event with no
    prompt neutrons.

    Returns
    -------
    n : int
    energies_eV : np.ndarray of float64, shape (n,)
    cosu, cosv, cosw : np.ndarray of float64, shape (n,)
    """
    cgmfwrap = _load_cgmfwrap()
    incident_energy_MeV = float(incident_energy_eV) * 1.0e-6

    for _ in range(_MAX_EVENT_RETRIES):
        event = cgmfwrap.run_event(int(zaid), incident_energy_MeV)
        n = int(event.nu_n)
        if n == 0:
            continue

        energies_eV = np.asarray(event.neutron_energies[:n], dtype=np.float64) * 1.0e6
        cosu = np.asarray(event.neutron_dir_cosu[:n], dtype=np.float64)
        cosv = np.asarray(event.neutron_dir_cosv[:n], dtype=np.float64)
        cosw = np.asarray(event.neutron_dir_cosw[:n], dtype=np.float64)
        return n, energies_eV, cosu, cosv, cosw

    raise RuntimeError(
        f"CGMF yielded no prompt neutron in {_MAX_EVENT_RETRIES} attempts "
        f"(zaid={int(zaid)}, E_in={float(incident_energy_eV):.3e} eV)."
    )


def fill_event(
    zaid,
    incident_energy_eV,
    n_out,
    energies_out_eV,
    cosu_out,
    cosv_out,
    cosw_out,
):
    """Sample one CGMF event and fill the provided buffers in place.

    Used from the Numba transport kernel through `objmode` so that the
    prompt-neutron buffer is allocated once per fission and the CGMF call
    happens at most twice (event + optional refill).

    Parameters
    ----------
    zaid : int
    incident_energy_eV : float
    n_out : np.ndarray of int64, shape (1,)
        Receives the realized prompt multiplicity, capped at the buffer size.
    energies_out_eV, cosu_out, cosv_out, cosw_out : np.ndarray of float64
        Receive the per-neutron data, sliced to n_out[0] valid entries.
    """
    n, energies_eV, cosu, cosv, cosw = run_event(zaid, incident_energy_eV)

    max_n = energies_out_eV.shape[0]
    if n > max_n:
        n = max_n

    n_out[0] = n
    energies_out_eV[:n] = energies_eV[:n]
    cosu_out[:n] = cosu[:n]
    cosv_out[:n] = cosv[:n]
    cosw_out[:n] = cosw[:n]
