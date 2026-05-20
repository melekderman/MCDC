import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation

with h5py.File("output_cgmf.h5", "r") as f1:
    E_cgmf     = f1["tallies/tracklength_tally_0/grid/energy"][:]
    t_cgmf     = f1["tallies/tracklength_tally_0/grid/time"][:]
    phi_cgmf   = f1["tallies/tracklength_tally_0/flux/mean"][:]      # shape: (Nt, NE)
    sd_cgmf    = f1["tallies/tracklength_tally_0/flux/sdev"][:]

with h5py.File("output_mcdc.h5", "r") as f2:
    E_mcdc     = f2["tallies/tracklength_tally_0/grid/energy"][:]
    t_mcdc     = f2["tallies/tracklength_tally_0/grid/time"][:]
    phi_mcdc   = f2["tallies/tracklength_tally_0/flux/mean"][:]    # shape: (Nt, NE)
    sd_mcdc    = f2["tallies/tracklength_tally_0/flux/sdev"][:]

E_mid_cgmf = 0.5 * (E_cgmf[:-1] + E_cgmf[1:])
E_mid_mcdc = 0.5 * (E_mcdc[:-1] + E_mcdc[1:])
t_mid_cgmf = 0.5 * (t_cgmf[:-1] + t_cgmf[1:])
t_mid_mcdc = 0.5 * (t_mcdc[:-1] + t_mcdc[1:])

# ── Figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# -- Right: φ(E) spectrum animation -------------------------------
ax = axes[1]
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Energy [eV]")
ax.set_ylabel(r"$\phi(E)$ [a.u.]")
ax.grid(True, which="both", ls="--", alpha=0.4)

print("phi_cgmf shape:", phi_cgmf.shape)
print("phi_mcdc shape:", phi_mcdc.shape)
print("t_mid_cgmf:", t_mid_cgmf.shape)
print("E_mid_cgmf:", E_mid_cgmf.shape)


E_1 = np.logspace(-4, 0, 50)                        # thermal: 1e-4 -> 1 eV
E_2 = np.logspace(0, 5, 200)                         # epithermal: 1 -> 1e5 eV
E_3 = np.logspace(5, np.log10(14e6), 100)          # fast: 1e5 -> 14 MeV

E_axis = np.concatenate([E_1, E_2[1:], E_3[1:]])

print("E_axis max:", E_axis[-1]) 

line_c,    = ax.plot([], [], "r-",  lw=1.5, label="CGMF")
fill_c     = ax.fill_between([], [], [], alpha=0.2, color="r")
line_m,    = ax.plot([], [], "b-",  lw=1.5, label="MC/DC")
fill_m     = ax.fill_between([], [], [], alpha=0.2, color="b")
time_text  = ax.set_title("")
ax.legend()

# Set y limits from full dataset
ymin = min(phi_cgmf[phi_cgmf > 0].min(), phi_mcdc[phi_mcdc > 0].min()) * 0.5
ymax = max(phi_cgmf.max(), phi_mcdc.max()) * 2.0
ax.set_ylim(ymin, ymax)
ax.set_xlim(E_mid_cgmf[0], E_mid_cgmf[-1])

# -- Left: integrated flux vs time (marker for current frame) ---
ax0 = axes[0]
phi_t_cgmf = phi_cgmf.sum(axis=0)   # integrate over energy
phi_t_sd_cgmf = np.sqrt((phi_sd_cgmf**2).sum(axis=0))
phi_t_mcdc = phi_mcdc.sum(axis=0)
phi_t_sd_mcdc = np.sqrt((phi_sd_mcdc**2).sum(axis=0))

ax0.plot(t_mid_cgmf * 1e9, phi_t_cgmf, "r-",  alpha=0.6, label="CGMF")
ax0.plot(t_mid_mcdc * 1e9, phi_t_mcdc, "b-",  alpha=0.6, label="MC/DC")
ax0.set_xlabel("$t$ [ns]")
ax0.set_ylabel("Integrated flux [a.u.]")
ax0.set_yscale("log")
ax0.legend()
ax0.grid(True, ls="--", alpha=0.4)
ax0.set_box_aspect(1)

(marker_c,) = ax0.plot([], [], "or", ms=8)
(marker_m,) = ax0.plot([], [], "ob", ms=8)

plt.tight_layout()
plt.subplots_adjust(top=0.88)

# ── Animate ───────────────────────────────────────────────────────
n_frames = min(len(t_mid_cgmf), len(t_mid_mcdc))

def animate(i):
    global fill_c, fill_m

    # Spectrum
    yc = phi_cgmf[:, i]
    ym = phi_mcdc[:, i]

    line_c.set_data(E_mid_cgmf, yc)
    line_m.set_data(E_mid_mcdc, ym)

    # Redraw fill_between
    fill_c.remove()
    fill_m.remove()
    fill_c = axes[1].fill_between(E_mid_cgmf, yc - sd_cgmf[:, i], yc + sd_cgmf[:, i],
                                   alpha=0.2, color="r")
    fill_m = axes[1].fill_between(E_mid_mcdc, ym - sd_mcdc[:, i], ym + sd_mcdc[:, i],
                                   alpha=0.2, color="b")

    axes[1].set_title(f"$t$ = {t_mid_cgmf[i]*1e9:.1f} ns")

    # Time marker
    marker_c.set_data([t_mid_cgmf[i] * 1e9], [phi_t_cgmf[i]])
    marker_m.set_data([t_mid_mcdc[i] * 1e9], [phi_t_mcdc[i]])

    return line_c, line_m, fill_c, fill_m, marker_c, marker_m

anim = animation.FuncAnimation(
    fig, animate, frames=n_frames, interval=80, blit=False
)

anim.save("cgmf_vs_mcdc_spectrum.gif", writer="pillow", fps=15)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

