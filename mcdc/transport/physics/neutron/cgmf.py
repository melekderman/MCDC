import importlib
import os
import sys


_CGMFWRAP = None
_MAX_EVENT_RETRIES = 64


def _load_cgmfwrap():
    global _CGMFWRAP

    if _CGMFWRAP is not None:
        return _CGMFWRAP

    path = os.environ.get("CGMFWRAP_PATH") or os.environ.get("CGMF")
    if path and path not in sys.path:
        sys.path.insert(0, path)

    try:
        _CGMFWRAP = importlib.import_module("cgmfwrap")
    except ImportError as exc:
        raise RuntimeError(
            "CGMF emission requested, but cgmfwrap could not be imported. "
            "Set CGMFWRAP_PATH or CGMF to the directory containing cgmfwrap."
        ) from exc

    return _CGMFWRAP


def sample_prompt_neutron(zaid, incident_energy_eV, xi):
    cgmfwrap = _load_cgmfwrap()
    incident_energy_MeV = float(incident_energy_eV) * 1.0e-6

    for _ in range(_MAX_EVENT_RETRIES):
        event = cgmfwrap.run_event(int(zaid), incident_energy_MeV)
        n = int(event.nu_n)

        idx = int(float(xi) * n)
        if idx >= n:
            idx = n - 1

        return (
            float(event.neutron_energies[idx]) * 1.0e6,
            float(event.neutron_dir_cosu[idx]),
            float(event.neutron_dir_cosv[idx]),
            float(event.neutron_dir_cosw[idx]),
        )

    raise RuntimeError("CGMF did not return any prompt neutron.")
