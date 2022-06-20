# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly over domains"""

import numpy as np
import pytest

import ufl
from dolfinx import cpp as _cpp
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         dirichletbc, form)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.mesh import (GhostMode, Mesh, create_unit_square, meshtags,
                          meshtags_from_entities, locate_entities_boundary,
                          locate_entities)

from mpi4py import MPI
from petsc4py import PETSc


@pytest.fixture
def mesh():
    return create_unit_square(MPI.COMM_WORLD, 10, 10)


def create_cell_meshtags_from_entities(mesh: Mesh, dim: int, cells: np.ndarray, values: np.ndarray):
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    cell_to_vertices = mesh.topology.connectivity(mesh.topology.dim, 0)
    entities = _cpp.graph.AdjacencyList_int32([cell_to_vertices.links(cell) for cell in cells])
    return meshtags_from_entities(mesh, dim, entities, values)


parametrize_ghost_mode = pytest.mark.parametrize("mode", [
    pytest.param(GhostMode.none, marks=pytest.mark.skipif(condition=MPI.COMM_WORLD.size > 1,
                                                          reason="Unghosted interior facets fail in parallel")),
    pytest.param(GhostMode.shared_facet, marks=pytest.mark.skipif(condition=MPI.COMM_WORLD.size == 1,
                                                                  reason="Shared ghost modes fail in serial"))])


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
@pytest.mark.parametrize("meshtags_factory", [meshtags, create_cell_meshtags_from_entities])
def test_assembly_dx_domains(mode, meshtags_factory):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Prepare a marking structures
    # indices cover all cells
    # values are [1, 2, 3, 3, ...]
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    indices = np.arange(0, num_cells)
    values = np.full(indices.shape, 3, dtype=np.intc)
    values[0] = 1
    values[1] = 2
    marker = meshtags_factory(mesh, mesh.topology.dim, indices, values)
    dx = ufl.Measure('dx', subdomain_data=marker, domain=mesh)
    w = Function(V)
    w.x.array[:] = 0.5

    # Assemble matrix
    a = form(w * ufl.inner(u, v) * (dx(1) + dx(2) + dx(3)))
    A = assemble_matrix(a)
    A.assemble()
    a2 = form(w * ufl.inner(u, v) * dx)
    A2 = assemble_matrix(a2)
    A2.assemble()
    assert (A - A2).norm() < 1.0e-12

    bc = dirichletbc(Function(V), range(30))

    # Assemble vector
    L = form(ufl.inner(w, v) * (dx(1) + dx(2) + dx(3)))
    b = assemble_vector(L)

    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                  mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    L2 = form(ufl.inner(w, v) * dx)
    b2 = assemble_vector(L2)
    apply_lifting(b2, [a], [[bc]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                   mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, [bc])
    assert (b - b2).norm() < 1.0e-12

    # Assemble scalar
    L = form(w * (dx(1) + dx(2) + dx(3)))
    s = assemble_scalar(L)
    s = mesh.comm.allreduce(s, op=MPI.SUM)
    assert s == pytest.approx(0.5, 1.0e-12)
    L2 = form(w * dx)
    s2 = assemble_scalar(L2)
    s2 = mesh.comm.allreduce(s2, op=MPI.SUM)
    assert s == pytest.approx(s2, 1.0e-12)


@pytest.mark.parametrize("mode", [GhostMode.none, GhostMode.shared_facet])
def test_assembly_ds_domains(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], 1.0)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    bottom_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, bottom)
    bottom_vals = np.full(bottom_facets.shape, 1, np.intc)

    top_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, top)
    top_vals = np.full(top_facets.shape, 2, np.intc)

    left_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, left)
    left_vals = np.full(left_facets.shape, 3, np.intc)

    right_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, right)
    right_vals = np.full(right_facets.shape, 6, np.intc)

    indices = np.hstack((bottom_facets, top_facets, left_facets, right_facets))
    values = np.hstack((bottom_vals, top_vals, left_vals, right_vals))

    indices, pos = np.unique(indices, return_index=True)
    marker = meshtags(mesh, mesh.topology.dim - 1, indices, values[pos])

    ds = ufl.Measure('ds', subdomain_data=marker, domain=mesh)

    w = Function(V)
    w.x.array[:] = 0.5

    bc = dirichletbc(Function(V), range(30))

    # Assemble matrix
    a = form(w * ufl.inner(u, v) * (ds(1) + ds(2) + ds(3) + ds(6)))
    A = assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()
    a2 = form(w * ufl.inner(u, v) * ds)
    A2 = assemble_matrix(a2)
    A2.assemble()
    norm2 = A2.norm()
    assert norm1 == pytest.approx(norm2, 1.0e-12)

    # Assemble vector
    L = form(ufl.inner(w, v) * (ds(1) + ds(2) + ds(3) + ds(6)))
    b = assemble_vector(L)

    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    L2 = form(ufl.inner(w, v) * ds)
    b2 = assemble_vector(L2)
    apply_lifting(b2, [a2], [[bc]])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b2, [bc])

    assert b.norm() == pytest.approx(b2.norm(), 1.0e-12)

    # Assemble scalar
    L = form(w * (ds(1) + ds(2) + ds(3) + ds(6)))
    s = assemble_scalar(L)
    s = mesh.comm.allreduce(s, op=MPI.SUM)
    L2 = form(w * ds)
    s2 = assemble_scalar(L2)
    s2 = mesh.comm.allreduce(s2, op=MPI.SUM)
    assert (s == pytest.approx(s2, 1.0e-12) and 2.0 == pytest.approx(s, 1.0e-12))


@parametrize_ghost_mode
def test_assembly_dS_domains(mode):
    N = 10
    mesh = create_unit_square(MPI.COMM_WORLD, N, N, ghost_mode=mode)
    one = Constant(mesh, PETSc.ScalarType(1))
    val = assemble_scalar(form(one * ufl.dS))
    val = mesh.comm.allreduce(val, op=MPI.SUM)
    assert val == pytest.approx(2 * (N - 1) + N * np.sqrt(2), 1.0e-7)


@parametrize_ghost_mode
def test_additivity(mode):
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12, ghost_mode=mode)
    V = FunctionSpace(mesh, ("Lagrange", 1))

    f1 = Function(V)
    f2 = Function(V)
    f3 = Function(V)
    f1.x.array[:] = 1.0
    f2.x.array[:] = 2.0
    f3.x.array[:] = 3.0
    j1 = ufl.inner(f1, f1) * ufl.dx(mesh)
    j2 = ufl.inner(f2, f2) * ufl.ds(mesh)
    j3 = ufl.inner(ufl.avg(f3), ufl.avg(f3)) * ufl.dS(mesh)

    # Assemble each scalar form separately
    J1 = mesh.comm.allreduce(assemble_scalar(form(j1)), op=MPI.SUM)
    J2 = mesh.comm.allreduce(assemble_scalar(form(j2)), op=MPI.SUM)
    J3 = mesh.comm.allreduce(assemble_scalar(form(j3)), op=MPI.SUM)

    # Sum forms and assemble the result
    J12 = mesh.comm.allreduce(assemble_scalar(form(j1 + j2)), op=MPI.SUM)
    J13 = mesh.comm.allreduce(assemble_scalar(form(j1 + j3)), op=MPI.SUM)
    J23 = mesh.comm.allreduce(assemble_scalar(form(j2 + j3)), op=MPI.SUM)
    J123 = mesh.comm.allreduce(assemble_scalar(form(j1 + j2 + j3)), op=MPI.SUM)

    # Compare assembled values
    assert (J1 + J2) == pytest.approx(J12)
    assert (J1 + J3) == pytest.approx(J13)
    assert (J2 + J3) == pytest.approx(J23)
    assert (J1 + J2 + J3) == pytest.approx(J123)


def test_manual_integration_domains():
    """Test that specifying integration domains manually i.e.
    by passing a list of cell indices or (cell, local facet) pairs
    to form gives the same result as the usual approach of tagging"""

    # NOTE Until https://github.com/FEniCS/dolfinx/pull/2244 is merged,
    # when assembling a matrix, only entries for exterior facets on the
    # exterior boundary will be added for ds integrals, and only entries
    # for interior facets on the interior boundary will be added for dS
    # interior facets.

    n = 4
    msh = create_unit_square(MPI.COMM_WORLD, n, n)

    V = FunctionSpace(msh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Create meshtags to mark some cells
    tdim = msh.topology.dim
    cell_map = msh.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts
    indices = np.arange(0, num_cells)
    values = np.zeros_like(indices, dtype=np.intc)
    marked_cells = locate_entities(
        msh, tdim, lambda x: x[0] < 0.75)
    values[marked_cells] = 7
    mt = meshtags(msh, tdim, indices, values)

    # Create meshtags to mark some exterior facets
    msh.topology.create_entities(tdim - 1)
    facet_map = msh.topology.index_map(tdim - 1)
    num_facets = facet_map.size_local + facet_map.num_ghosts
    facet_indices = np.arange(0, num_facets)
    facet_values = np.zeros_like(facet_indices, dtype=np.intc)
    marked_ext_facets = locate_entities_boundary(
        msh, tdim - 1, lambda x: np.isclose(x[0], 0.0))
    marked_int_facets = locate_entities(
        msh, tdim - 1, lambda x: x[0] < 0.75)
    # marked_int_facets will also contain facets on the boundary,
    # so set these values first, followed by the values for
    # marked_ext_facets
    facet_values[marked_int_facets] = 3
    facet_values[marked_ext_facets] = 6
    mt_facets = meshtags(msh, tdim - 1, facet_indices, facet_values)

    # Create measures
    dx_mt = ufl.Measure("dx", subdomain_data=mt, domain=msh)
    ds_mt = ufl.Measure("ds", subdomain_data=mt_facets, domain=msh)
    dS_mt = ufl.Measure("dS", subdomain_data=mt_facets, domain=msh)

    # Create a forms and assemble
    L = form(ufl.inner(1.0, v) * (dx_mt(7) + ds_mt(6))
             + ufl.inner(1.0, v("+") + v("-")) * dS_mt(3))
    b = assemble_vector(L)
    b_expected_norm = b.norm()

    a = form(ufl.inner(u, v) * (dx_mt(7) + ds_mt(6))
             + ufl.inner(u("+"), v("+") + v("-")) * dS_mt(3))
    A = assemble_matrix(a)
    A.assemble()
    A_expected_norm = A.norm()

    # Manually specify cells to integrate over (need to remove ghosts
    # to give same result as above)
    cell_domain = [c for c in marked_cells if c < cell_map.size_local]

    # Manually specify exterior facets to integrate over as
    # (cell, local facet) pairs
    ext_facet_domain = []
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    c_to_f = msh.topology.connectivity(tdim, tdim - 1)
    f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    for f in marked_ext_facets:
        if f < facet_map.size_local:
            c = f_to_c.links(f)[0]
            local_f = np.where(c_to_f.links(c) == f)[0][0]
            ext_facet_domain.append(c)
            ext_facet_domain.append(local_f)

    # Manually specify interior facets to integrate over
    int_facet_domain = []
    for f in marked_int_facets:
        if f >= facet_map.size_local or len(f_to_c.links(f)) != 2:
            continue

        c_0, c_1 = f_to_c.links(f)[0], f_to_c.links(f)[1]
        local_f_0 = np.where(c_to_f.links(c_0) == f)[0][0]
        local_f_1 = np.where(c_to_f.links(c_1) == f)[0][0]

        int_facet_domain.append(c_0)
        int_facet_domain.append(local_f_0)
        int_facet_domain.append(c_1)
        int_facet_domain.append(local_f_1)

    # Create measures
    cell_domains = {7: cell_domain}
    dx_manual = ufl.Measure("dx", subdomain_data=cell_domains, domain=msh)

    ext_facet_domains = {6: ext_facet_domain}
    ds_manual = ufl.Measure("ds", subdomain_data=ext_facet_domains, domain=msh)

    int_facet_domains = {3: int_facet_domain}
    dS_manual = ufl.Measure("dS", subdomain_data=int_facet_domains, domain=msh)

    # Assemble forms and check
    L = form(ufl.inner(1.0, v) * (dx_manual(7) + ds_manual(6))
             + ufl.inner(1.0, v("+") + v("-")) * dS_manual(3))
    b = assemble_vector(L)
    b_norm = b.norm()

    assert(np.isclose(b_norm, b_expected_norm))

    a = form(ufl.inner(u, v) * (dx_manual(7) + ds_manual(6))
             + ufl.inner(u("+"), v("+") + v("-")) * dS_manual(3))
    A = assemble_matrix(a)
    A.assemble()
    A_norm = A.norm()

    assert(np.isclose(A_norm, A_expected_norm))


# TODO Parametrise for ghost mode
def test_ext_facet_perms():
    # NOTE N must be even
    n = 2

    # NOTE No permutations should be needed even on a random mesh,
    # since everything belongs to a single cell
    msh = create_unit_square(
        MPI.COMM_WORLD, n, n, ghost_mode=GhostMode.none)

    V = FunctionSpace(msh, ("Lagrange", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    tdim = msh.topology.dim
    left_cells = locate_entities(
        msh, tdim, lambda x: x[0] <= 0.5)
    centre_facets = locate_entities(
        msh, tdim - 1, lambda x: np.isclose(x[0], 0.5))
    left_boundary_facets = locate_entities_boundary(
        msh, tdim - 1, lambda x: np.isclose(x[0], 0.0))

    # from dolfinx.mesh import create_submesh
    # left_cell_submesh = create_submesh(msh, tdim, left_cells)[0]
    # entity_submesh = create_submesh(msh, tdim - 1, marked_facets)[0]
    # from dolfinx.io import XDMFFile
    # with XDMFFile(msh.comm, "msh.xdmf", "w") as file:
    #     file.write_mesh(msh)
    # with XDMFFile(msh.comm, "left_cells.xdmf", "w") as file:
    #     file.write_mesh(left_cell_submesh)
    # with XDMFFile(msh.comm, "entities.xdmf", "w") as file:
    #     file.write_mesh(entity_submesh)

    # Manually specify exterior facets to integrate over as
    # (cell, local facet) pairs
    left_cell_ext_facet_domain = []
    right_cell_ext_facet_domain = []
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    c_to_f = msh.topology.connectivity(tdim, tdim - 1)
    f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    facet_map = msh.topology.index_map(tdim - 1)
    cell_map = msh.topology.index_map(tdim)
    for f in centre_facets:
        for c in f_to_c.links(f):
            if c < cell_map.size_local:
                local_f = np.where(c_to_f.links(c) == f)[0][0]

                if c in left_cells:
                    left_cell_ext_facet_domain.append(c)
                    left_cell_ext_facet_domain.append(local_f)
                else:
                    right_cell_ext_facet_domain.append(c)
                    right_cell_ext_facet_domain.append(local_f)

    ds_left = ufl.Measure(
        "ds", subdomain_data={1: left_cell_ext_facet_domain}, domain=msh)
    ds_right = ufl.Measure(
        "ds", subdomain_data={1: right_cell_ext_facet_domain}, domain=msh)

    num_facets = facet_map.size_local + facet_map.num_ghosts
    indices = np.arange(0, num_facets)
    values = np.zeros_like(indices, dtype=np.intc)
    values[left_boundary_facets] = 1
    marker = meshtags(msh, msh.topology.dim - 1, indices, values)
    ds = ufl.Measure("ds", subdomain_data=marker, domain=msh)

    f = Function(V)
    f.interpolate(lambda x: x[1]**2)

    s = msh.comm.allreduce(
        assemble_scalar(form(f * ds(1))), op=MPI.SUM)
    s_left = msh.comm.allreduce(
        assemble_scalar(form(f * ds_left(1))), op=MPI.SUM)
    s_right = msh.comm.allreduce(
        assemble_scalar(form(f * ds_right(1))), op=MPI.SUM)

    assert(np.isclose(s, s_left))
    assert(np.isclose(s, s_right))

    L = form(ufl.inner(f, v) * ds(1))
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                  mode=PETSc.ScatterMode.REVERSE)

    L_left = form(ufl.inner(f, v) * ds_left(1))
    b_left = assemble_vector(L_left)
    b_left.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)

    L_right = form(ufl.inner(f, v) * ds_right(1))
    b_right = assemble_vector(L_right)
    b_right.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)

    assert(np.isclose(b.norm(), b_left.norm()))
    assert(np.isclose(b.norm(), b_right.norm()))

    a = form(ufl.inner(f * u, v) * ds(1))
    A = assemble_matrix(a)
    A.assemble()

    a_left = form(ufl.inner(f * u, v) * ds_left(1))
    A_left = assemble_matrix(a_left)
    A_left.assemble()

    a_right = form(ufl.inner(f * u, v) * ds_right(1))
    A_right = assemble_matrix(a_right)
    A_right.assemble()

    print()
    print(A.norm(), A_left.norm(), A_right.norm())

    assert(np.isclose(A.norm(), A_left.norm()))
    assert(np.isclose(A.norm(), A_right.norm()))
