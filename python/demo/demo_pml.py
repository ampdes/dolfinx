import sys

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)
import numpy as np

try:
    import pyvista
    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False
from data.mesh_wire_pml import generate_mesh_wire

import ufl
from data.utils import calculate_analytical_efficiencies
from dolfinx import fem, plot
from dolfinx.io import VTXWriter
from dolfinx.io.gmshio import model_to_mesh
from ufl import FacetNormal, as_matrix, as_vector, cross, curl, inner, lhs, rhs, sqrt, algebra, \
    transpose, det, inv, grad

from mpi4py import MPI
from petsc4py import PETSc

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)

class BackgroundElectricField:

    def __init__(self, theta, n_b, k0):
        self.theta = theta
        self.k0 = k0
        self.n_b = n_b

    def eval(self, x):

        kx = self.n_b * self.k0 * np.cos(self.theta)
        ky = self.n_b * self.k0 * np.sin(self.theta)
        phi = kx * x[0] + ky * x[1]

        ax = np.sin(self.theta)
        ay = np.cos(self.theta)

        return (-ax * np.exp(1j * phi), ay * np.exp(1j * phi))

def curl_2d(a):
    """Returns the curl of two 2D vectors as a 3D vector"""
    ay_x = a[1].dx(0)
    ax_y = a[0].dx(1)
    return as_vector((0, 0, ay_x - ax_y))

def pml_coordinates(x, alpha, k0, l_dom, l_pml):

    # Define boolean functions for the PML regions
    inside_pml_x = (ufl.sign(ufl.sign(x[0])*x[0] - l_dom/2) + 1)/2
    inside_pml_y = (ufl.sign(ufl.sign(x[1])*x[1] - l_dom/2) + 1)/2

    # Define the coordinate transformation for PML regions
    x_pml = x[0] + 1j*alpha/k0*x[0]*(ufl.sign(x[0])*x[0] - l_dom/2)/(l_pml/2 - l_dom/2)**2*inside_pml_x
    y_pml = x[1] + 1j*alpha/k0*x[1]*(ufl.sign(x[1])*x[1] - l_dom/2)/(l_pml/2 - l_dom/2)**2*inside_pml_y

    return as_vector((x_pml, y_pml))

um = 10**-6  # micron
nm = um*10**-3  # nanometer
epsilon_0 = 8.8541878128 * 10**-12
mu_0 = 4 * np.pi * 10**-7

# Radius of the wire and of the boundary of the domain
radius_wire = 0.05 * um
l_dom = 0.4 * um
l_pml = 0.6 * um

# The smaller the mesh_factor, the finer is the mesh
mesh_factor = 1

# Mesh size inside the wire
in_wire_size = mesh_factor * 6 * nm

# Mesh size at the boundary of the wire
on_wire_size = mesh_factor * 3 * nm

# Mesh size in the background
bkg_size = mesh_factor * 20 * nm

# Mesh size at the boundary
pml_size = mesh_factor * 20 * nm

# Tags for the subdomains
au_tag = 1
bkg_tag = 2
pml_tag = 3
# -

model = generate_mesh_wire(
    radius_wire, l_dom, l_pml, in_wire_size, on_wire_size, bkg_size,
    pml_size, au_tag, bkg_tag, pml_tag)

mesh, cell_tags, facet_tags = model_to_mesh(
    model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()
MPI.COMM_WORLD.barrier()
# -

wl0 = 0.4 * um  # Wavelength of the background field
n_bkg = 1  # Background refractive index
eps_bkg = n_bkg**2  # Background relative permittivity
k0 = 2 * np.pi / wl0  # Wavevector of the background field
deg = np.pi / 180
theta = 0 * deg  # Angle of incidence of the background field

degree = 3
curl_el = ufl.FiniteElement("N1curl", mesh.ufl_cell(), degree)
V = fem.FunctionSpace(mesh, curl_el)

f = BackgroundElectricField(theta, n_bkg, k0)
Eb = fem.Function(V)
Eb.interpolate(f.eval)

x = ufl.SpatialCoordinate(mesh)

alpha = 1
r_pml = pml_coordinates(x, alpha, k0, l_dom, l_pml)
J_pml = grad(r_pml)

# Transform the 2x2 Jacobian into a 3x3 matrix.
J_pml = as_matrix(((J_pml[0, 0], 0          , 0),
                   (0          , J_pml[1, 1], 0),
                   (0          , 0          , 1)))

A_pml = inv(J_pml)
pml_matrix = det(J_pml) * A_pml * transpose(A_pml)
eps_pml = eps_bkg * pml_matrix
mu_pml = inv(pml_matrix)

Es = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

Es_3d = as_vector((Es[0], Es[1], 0))
v_3d = as_vector((v[0], v[1], 0))

dx = ufl.Measure("dx", mesh, subdomain_data=cell_tags)
dAu = dx(au_tag)
dBkg = dx(bkg_tag)
dDom = dAu + dBkg
dPml = dx(pml_tag)

reps_au = -1.0782
ieps_au = 5.8089
eps_au = reps_au + ieps_au * 1j

D = fem.FunctionSpace(mesh, ("DG", 0))
eps = fem.Function(D)
au_cells = cell_tags.find(au_tag)
bkg_cells = cell_tags.find(bkg_tag)
eps.x.array[au_cells] = np.full_like(
    au_cells, reps_au + ieps_au * 1j, dtype=np.complex128)
eps.x.array[bkg_cells] = np.full_like(bkg_cells, eps_bkg, dtype=np.complex128)
eps.x.scatter_forward()

F = - inner(curl_2d(Es), curl_2d(v)) * dDom \
    + eps * k0 ** 2 * inner(Es, v) * dDom \
    + k0 ** 2 * (eps - eps_bkg) * inner(Eb, v) * dDom \
    - inner(mu_pml * curl_2d(Es), curl_2d(v)) * dPml \
    + k0 ** 2 * inner(eps_pml * Es_3d, v_3d) * dPml

a, L = lhs(F), rhs(F)

problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu"})
Esh = problem.solve()

V_dg = fem.VectorFunctionSpace(mesh, ("DG", degree))
Esh_dg = fem.Function(V_dg)
Esh_dg.interpolate(Esh)

with VTXWriter(MPI.COMM_WORLD, "Esh.bp", Esh_dg) as f:
    f.write(0.0)

E = fem.Function(V)
E.x.array[:] = Eb.x.array[:] + Esh.x.array[:]

E_dg = fem.Function(V_dg)
E_dg.interpolate(E)

with VTXWriter(MPI.COMM_WORLD, "E.bp", E_dg) as f:
    f.write(0.0)

q_abs_analyt, q_sca_analyt, q_ext_analyt = calculate_analytical_efficiencies(
    reps_au,
    ieps_au,
    n_bkg,
    wl0,
    radius_wire)

Z0 = np.sqrt(mu_0 / epsilon_0)

E_3d = as_vector((E[0], E[1], 0))

I0 = 0.5 / Z0

gcs = 2 * radius_wire

Q = 0.5 * ieps_au * k0 * (inner(E_3d, E_3d)) / Z0 / n_bkg

q_abs_fenics_proc = (fem.assemble_scalar(fem.form(Q * dAu)) / gcs / I0).real

q_abs_fenics = mesh.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

err_abs = np.abs(q_abs_analyt - q_abs_fenics) / q_abs_analyt * 100

if MPI.COMM_WORLD.rank == 0:

    print()
    print(f"The analytical absorption efficiency is {q_abs_analyt}")
    print(f"The numerical absorption efficiency is {q_abs_fenics}")
    print(f"The error is {err_abs}%")
    print()
# -