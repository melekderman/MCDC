## MCNP vs MCDC examples
# U-235 Sphere
import mcdc
import numpy as np
import os

os.environ["MCDC_LIB"] = "../mcdc_data"

#===========
# Set Model
#===========

U_235 = mcdc.Material(nuclide_composition={"U235": 0.048807514})

# Set Surfaces
# 
sphere = mcdc.Surface.Sphere(center=[0, 0, 0], radius=5.0, boundary_condition="vacuum")

inside_sphere = -sphere
sphere_cell = mcdc.Cell(region=inside_sphere, fill=U_235)

# Set Source
ENERGY = 14e6
energy = np.array([[ENERGY - 1e-8, ENERGY + 1e-8], [0.5,0.5]])
mcdc.Source(position=[0,0,0], isotropic=True, energy=energy) # energy in ev


r = np.linspace(0,5,100)
theta = np.linspace(0,np.pi,100)
phi = np.linspace(0,2*np.pi,100)

x = r *np.cos(theta) * np.sin(phi)
y = r *np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
mesh = np.meshgrid(x,y,z)

E_1 = np.linspace(1e6,14e6,100)
#E_2 = np.linspace(200,1e5,1000)
#E_3 = np.linspace(1.1e5,14e6,1000)
#E_axis = np.concatenate([E_1, E_2, E_3])
E_axis = E_1

# tallies

# whole sphere
mcdc.Tally(cell=sphere_cell, scores=["flux"],energy=E_axis)

# Settings
N = 10

mcdc.settings.N_particle = N
mcdc.settings.active_bank_buffer = 100*N
mcdc.settings.set_fission_emission_model("cgmf")

mcdc.run()
