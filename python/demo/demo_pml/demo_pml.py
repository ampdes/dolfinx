# # Scattering from a wire with perfectly matched layer condition
# This demo is implemented in three files: one for the mesh
# generation with gmsh, one for the calculation of analytical efficiencies,
# and one for the variational forms and the solver. It illustrates how to:
#
# - Use complex quantities in FEniCSx
# - Setup and solve Maxwell's equations
# - Implement (rectangular) perfectly matched layers
#
# ## Equations, problem definition and implementation
#
# First of all, let's import the modules that will be used:

# +
import sys
from matplotlib.cbook import to_filehandle

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
from analytical_efficiencies_wire import calculate_analytical_efficiencies
from mesh_wire_pml import generate_mesh_wire

import ufl
from dolfinx import fem, plot
from dolfinx.io import VTXWriter
from dolfinx.io.gmshio import model_to_mesh
from ufl import (FacetNormal, algebra, as_matrix, as_vector, cross, curl, det,
                 grad, inner, inv, lhs, rhs, sqrt, transpose)

from mpi4py import MPI
from petsc4py import PETSc

# -

# Since we want to solve time-harmonic Maxwell's equation, we need to
# specify that the demo should only be executed with DOLFINx complex mode,
# otherwise it would not work:

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)

# Now, let's consider an infinite metallic wire immersed in
# a background medium (e.g. vacuum or water). Let's now
# consider the plane cutting the wire perpendicularly to
# its axis at a generic point. Such plane $\Omega=\Omega_{m}
# \cup\Omega_{b}$ is formed by the cross-section
# of the wire $\Omega_m$ and the background medium
# $\Omega_{b}$ surrounding the wire. We want to calculate
# the electric field $\mathbf{E}_s$ scattered by the wire
# when a background wave $\mathbf{E}_b$ impinges on it.
# We will consider a background plane wave at $\lambda_0$
# wavelength, which can be written analytically as:
#
# $$
# \mathbf{E}_b = \exp(\mathbf{k}\cdot\mathbf{r})\hat{\mathbf{u}}_p
# $$
#
# with $\mathbf{k} = \frac{2\pi}{\lambda_0}n_b\hat{\mathbf{u}}_k$
# being the wavevector of the
# plane wave, pointing along the propagation direction,
# with $\hat{\mathbf{u}}_p$ being the
# polarization direction, and with $\mathbf{r}$ being a
# point in $\Omega$.
# We will only consider $\hat{\mathbf{u}}_k$ and $\hat{\mathbf{u}}_p$
# with components belonging
# to the $\Omega$ domain and perpendicular to each other,
# i.e. $\hat{\mathbf{u}}_k \perp \hat{\mathbf{u}}_p$
# (transversality condition of plane waves).
# If we call $x$ and $y$ the horizontal
# and vertical axis in our $\Omega$ domain,
# and by defining $k_x = n_bk_0\cos\theta$ and
# $k_y = n_bk_0\sin\theta$, with $\theta$ being the angle
# defined by the propagation direction $\hat{\mathbf{u}}_k$
# and the horizontal axis $\hat{\mathbf{u}}_x$,
# we can write more explicitly:
#
# $$
# \mathbf{E}_b = -\sin\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_x
# + \cos\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_y
# $$
#
# The `BackgroundElectricField` class below implements such function.
# The inputs to the function are the angle $\theta$, the background
# refractive index $n_b$ and the vacuum wavevector $k_0$. The
# function returns the expression $ \mathbf{E}_b = -\sin
# \theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_x
# + \cos\theta e^{j (k_xx+k_yy)}\hat{\mathbf{u}}_y$.

# +

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

# -

# Let's now define the $\nabla\times$ operator for 2d vector, since
# we will use it later:

# +

def curl_2d(a):

    return as_vector((0, 0, a[1].dx(0) - a[0].dx(1)))

# -

# As said before, we are going to implement a perfectly matched layer (PML)
# in this problem. PMLs are reflectionless layers surrounding the 
# domain that gradually absorb waves impinging on them, and
# therefore are used for truncating the domain. To implement them, we
# can use a complex transformation of coordinate in the PML domain. For
# squared PML we can define this transformation as:
#
# \begin{align}
# & x^\prime= x\left\{1-j\frac{\alpha}{k_0}\left[\frac{|x|-l_{dom}/2}
# {(l_{pml}/2 - l_{dom}/2)^2}\right] \right\}\\
# & y^\prime= y\left\{1-j\frac{\alpha}{k_0}\left[\frac{|y|-l_{dom}/2}
# {(l_{pml}/2 - l_{dom}/2)^2}\right] \right\}\\
# \end{align}
#
# with $l_{dom}$ and $l_{pml}$ being the lengths of the domain 
# without and with PML, respectively, and with $\alpha$ being a parameter
# that tunes the absorption within the PML (the greater the $\alpha$,
# the faster the absorption). In DOLFINx, we can define this
# coordinate transformation in the following way:

def pml_coordinates(x, alpha, k0, l_dom, l_pml):

    inside_pml = [(ufl.sign(ufl.sign(x[i]) * x[i] - l_dom / 2) + 1) / 2 for i in range(len(x))]

    return as_vector([x[i] + 1j * alpha / k0 * x[i] * (ufl.sign(x[i]) * x[i] - l_dom / 2) / 
                    (l_pml / 2 - l_dom / 2)**2 * inside_pml[i] for i in range(len(x))])


# The `inside_pml` function is a boolean that switch on the transformation
# inside the PML region.
#
# Next we define some mesh specific parameters:

um = 10**-6  # micron
nm = um * 10**-3  # nanometer
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

# We generate the mesh using GMSH and convert it to a
# `dolfinx.mesh.Mesh`.

model = generate_mesh_wire(
    radius_wire, l_dom, l_pml, in_wire_size, on_wire_size, bkg_size,
    pml_size, au_tag, bkg_tag, pml_tag)

mesh, cell_tags, facet_tags = model_to_mesh(
    model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()
MPI.COMM_WORLD.barrier()
# -

# Let's have a visual check of the mesh by plotting it with PyVista:

if have_pyvista:
    topology, cell_types, geometry = plot.create_vtk_mesh(mesh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    pyvista.set_jupyter_backend("pythreejs")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        figure = plotter.screenshot("wire_mesh.png")

# Now we define some other problem specific parameters:

wl0 = 0.4 * um  # Wavelength of the background field
n_bkg = 1  # Background refractive index
eps_bkg = n_bkg**2  # Background relative permittivity
k0 = 2 * np.pi / wl0  # Wavevector of the background field
deg = np.pi / 180
theta = 0 * deg  # Angle of incidence of the background field

# And then the function space used for the electric field.
# We will use a 3rd order
# [Nedelec (first kind)](https://defelement.com/elements/nedelec1.html)
# element:
#

degree = 3
curl_el = ufl.FiniteElement("N1curl", mesh.ufl_cell(), degree)
V = fem.FunctionSpace(mesh, curl_el)

# Next, we interpolate $\mathbf{E}_b$ into the function space $V$:

f = BackgroundElectricField(theta, n_bkg, k0)
Eb = fem.Function(V)
Eb.interpolate(f.eval)

# Definition of Trial and Test functions
Es = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Measures for subdomains
dx = ufl.Measure("dx", mesh, subdomain_data=cell_tags)
dDom = dx((au_tag, bkg_tag))
dPml = dx(pml_tag)


# Now it is the turn of the permittivity $\varepsilon$.
# First of all let's define the relative permittivity $\varepsilon_m$
# of the gold wire at $400nm$ (data taken from
# [*Olmon et al. 2012*](https://doi.org/10.1103/PhysRevB.86.235147)
# , and for a quick reference have a look at [refractiveindex.info](
# https://refractiveindex.info/?shelf=main&book=Au&page=Olmon-sc
# )):

# Definition of relative permittivity for Au @400nm
eps_au = -1.0782 + 1j * 5.8089


# We want to define a space function for the permittivity
# $\varepsilon$ that takes the value of the gold permittivity $\varepsilon_m$
# for cells inside the wire, while it takes the value of the
# background permittivity otherwise:

D = fem.FunctionSpace(mesh, ("DG", 0))
eps = fem.Function(D)
au_cells = cell_tags.find(au_tag)
bkg_cells = cell_tags.find(bkg_tag)
eps.x.array[au_cells] = np.full_like(
    au_cells, eps_au, dtype=np.complex128)
eps.x.array[bkg_cells] = np.full_like(bkg_cells, eps_bkg, dtype=np.complex128)
eps.x.scatter_forward()

# Now we need to define our weak form in DOLFINx. For the PML,
# we can 

x = ufl.SpatialCoordinate(mesh)

alpha = 1
r_pml = pml_coordinates(x, alpha, k0, l_dom, l_pml)
J_pml = grad(r_pml)

# Transform the 2x2 Jacobian into a 3x3 matrix.
J_pml = as_matrix(((J_pml[0, 0], 0, 0),
                   (0, J_pml[1, 1], 0),
                   (0, 0, 1)))

A_pml = inv(J_pml)
pml_matrix = det(J_pml) * A_pml * transpose(A_pml)
eps_pml = eps_bkg * pml_matrix
mu_pml = inv(pml_matrix)

Es_3d = as_vector((Es[0], Es[1], 0))
v_3d = as_vector((v[0], v[1], 0))

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

with VTXWriter(mesh.comm, "Esh.bp", Esh_dg) as f:
    f.write(0.0)

E = fem.Function(V)
E.x.array[:] = Eb.x.array[:] + Esh.x.array[:]

E_dg = fem.Function(V_dg)
E_dg.interpolate(E)

with VTXWriter(mesh.comm, "E.bp", E_dg) as f:
    f.write(0.0)

q_abs_analyt, q_sca_analyt, q_ext_analyt = calculate_analytical_efficiencies(
    eps_au,
    n_bkg,
    wl0,
    radius_wire)

Z0 = np.sqrt(mu_0 / epsilon_0)

E_3d = as_vector((E[0], E[1], 0))

I0 = 0.5 / Z0

gcs = 2 * radius_wire

Q = 0.5 * eps_au.imag * k0 * (inner(E_3d, E_3d)) / Z0 / n_bkg

dAu = dx(au_tag)

q_abs_fenics_proc = (fem.assemble_scalar(fem.form(Q * dAu)) / gcs / I0).real

q_abs_fenics = mesh.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

err_abs = np.abs(q_abs_analyt - q_abs_fenics) / q_abs_analyt

# Check if error is less than 1%
assert err_abs < 0.01

if MPI.COMM_WORLD.rank == 0:

    print()
    print(f"The analytical absorption efficiency is {q_abs_analyt}")
    print(f"The numerical absorption efficiency is {q_abs_fenics}")
    print(f"The error is {err_abs*100}%")
    print()
