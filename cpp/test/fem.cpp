// Copyright (C) 2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include <algorithm>
#include <catch2/catch.hpp>
#include <concepts>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <mpi.h>

using namespace dolfinx;

namespace
{

template <std::floating_point E, std::floating_point T>
void test_function_space()
{
  auto mesh = std::make_shared<mesh::Mesh<T>>(
      mesh::create_rectangle<T>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                {22, 12}, mesh::CellType::triangle));

  // Create a Basix continuous Lagrange element of degree 1
  basix::FiniteElement e = basix::create_element<E>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<E, T>>(
      fem::create_functionspace(mesh, e, 1));

  auto x = V->tabulate_dof_coordinates(false);
}

} // namespace

TEST_CASE("Element creation and types")
{
  CHECK_NOTHROW(test_function_space<float, float>());
  CHECK_NOTHROW(test_function_space<double, double>());
  CHECK_NOTHROW(test_function_space<double, float>());
  CHECK_NOTHROW(test_function_space<float, double>());
}

#endif
