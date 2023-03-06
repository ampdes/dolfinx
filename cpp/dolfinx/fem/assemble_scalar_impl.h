// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "Form.h"
#include "FunctionSpace.h"
#include "utils.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <memory>
#include <vector>

namespace dolfinx::fem::impl
{

/// Assemble functional over cells
template <typename T>
T assemble_cells(const mesh::Geometry<scalar_value_type_t<T>>& geometry,
                 std::span<const std::int32_t> cells, FEkernel<T> auto fn,
                 std::span<const T> constants, std::span<const T> coeffs,
                 int cstride)
{
  T value(0);
  if (cells.empty())
    return value;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  auto x = geometry.x();

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * num_dofs_g);

  // Iterate over all cells
  for (std::size_t index = 0; index < cells.size(); ++index)
  {
    std::int32_t c = cells[index];

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    const T* coeff_cell = coeffs.data() + index * cstride;
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(), nullptr,
       nullptr);
  }

  return value;
}

/// Execute kernel over exterior facets and accumulate result
template <typename T>
T assemble_exterior_facets(
    const mesh::Geometry<scalar_value_type_t<T>>& geometry, int num_cell_facets,
    std::span<const std::int32_t> facets, FEkernel<T> auto fn,
    std::span<const T> constants, std::span<const T> coeffs, int cstride,
    std::span<const std::uint8_t> perms,
    const std::function<std::uint8_t(std::int32_t, std::int32_t)>&
        get_facet_perm)
{
  T value(0);
  if (facets.empty())
    return value;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  auto x = geometry.x();

  // Create data structures used in assembly
  std::vector<scalar_value_type_t<T>> coordinate_dofs(3 * num_dofs_g);

  // Iterate over all facets
  assert(facets.size() % 2 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 2)
  {
    std::int32_t cell = facets[index];
    std::int32_t local_facet = facets[index + 1];

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t i = 0; i < x_dofs.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs[i]), 3,
                  std::next(coordinate_dofs.begin(), 3 * i));
    }

    const T* coeff_cell = coeffs.data() + index / 2 * cstride;
    const std::array<std::uint8_t, 2> perm
        = {perms[cell * num_cell_facets + local_facet],
           get_facet_perm(cell, local_facet)};
    fn(&value, coeff_cell, constants.data(), coordinate_dofs.data(),
       &local_facet, perm.data());
  }

  return value;
}

/// Assemble functional over interior facets
template <typename T>
T assemble_interior_facets(
    const mesh::Geometry<scalar_value_type_t<T>>& geometry, int num_cell_facets,
    std::span<const std::int32_t> facets, FEkernel<T> auto fn,
    std::span<const T> constants, std::span<const T> coeffs, int cstride,
    std::span<const int> offsets, std::span<const std::uint8_t> perms)
{
  T value(0);
  if (facets.empty())
    return value;

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  const std::size_t num_dofs_g = geometry.cmap().dim();
  auto x = geometry.x();

  // Create data structures used in assembly
  using X = scalar_value_type_t<T>;
  std::vector<X> coordinate_dofs(2 * num_dofs_g * 3);
  std::span<X> cdofs0(coordinate_dofs.data(), num_dofs_g * 3);
  std::span<X> cdofs1(coordinate_dofs.data() + num_dofs_g * 3, num_dofs_g * 3);

  std::vector<T> coeff_array(2 * offsets.back());
  assert(offsets.back() == cstride);

  // Iterate over all facets
  assert(facets.size() % 4 == 0);
  for (std::size_t index = 0; index < facets.size(); index += 4)
  {
    std::array<std::int32_t, 2> cells = {facets[index], facets[index + 2]};
    std::array<std::int32_t, 2> local_facet
        = {facets[index + 1], facets[index + 3]};

    // Get cell geometry
    auto x_dofs0 = x_dofmap.links(cells[0]);
    for (std::size_t i = 0; i < x_dofs0.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs0[i]), 3,
                  std::next(cdofs0.begin(), 3 * i));
    }
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (std::size_t i = 0; i < x_dofs1.size(); ++i)
    {
      std::copy_n(std::next(x.begin(), 3 * x_dofs1[i]), 3,
                  std::next(cdofs1.begin(), 3 * i));
    }

    const std::array perm{perms[cells[0] * num_cell_facets + local_facet[0]],
                          perms[cells[1] * num_cell_facets + local_facet[1]]};
    fn(&value, coeffs.data() + index / 2 * cstride, constants.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data());
  }

  return value;
}

/// Assemble functional into an scalar with provided mesh geometry.
template <typename T>
T assemble_scalar(
    const fem::Form<T>& M,
    const mesh::Geometry<scalar_value_type_t<T>>& geometry,
    std::span<const T> constants,
    const std::map<std::pair<IntegralType, int>,
                   std::pair<std::span<const T>, int>>& coefficients)
{
  std::shared_ptr<const mesh::Mesh> mesh = M.mesh();
  assert(mesh);

  T value = 0;
  for (int i : M.integral_ids(IntegralType::cell))
  {
    const auto& fn = M.kernel(IntegralType::cell, i);
    const auto& [coeffs, cstride] = coefficients.at({IntegralType::cell, i});
    const std::vector<std::int32_t>& cells = M.cell_domains(i);
    value += impl::assemble_cells(geometry, cells, fn, constants, coeffs,
                                  cstride);
  }

  mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint8_t>& perms
      = mesh->topology().get_facet_permutations();

  mesh->topology_mutable().create_connectivity(mesh->topology().dim(),
                                               mesh->topology().dim() - 1);
  // TODO Package as (cell, local_facet) pairs in
  // create_full_cell_permutations?
  auto c_to_f = mesh->topology().connectivity(mesh->topology().dim(),
                                              mesh->topology().dim() - 1);
  mesh->topology_mutable().create_full_cell_permutations();
  const std::vector<std::uint8_t>& facet_perms
      = mesh->topology().get_full_cell_permutations();
  std::function<std::uint8_t(std::int32_t, std::int32_t)> get_facet_perm
      = [&facet_perms, c_to_f](std::int32_t c, std::int32_t local_f)
  { return facet_perms[c_to_f->links(c)[local_f]]; };
  int num_cell_facets = mesh::cell_num_entities(mesh->topology().cell_type(),
                                                mesh->topology().dim() - 1);

  for (int i : M.integral_ids(IntegralType::exterior_facet))
  {
    const auto& fn = M.kernel(IntegralType::exterior_facet, i);
    const auto& [coeffs, cstride]
        = coefficients.at({IntegralType::exterior_facet, i});
    const std::vector<std::int32_t>& facets = M.exterior_facet_domains(i);
    value += impl::assemble_exterior_facets(geometry, num_cell_facets, constants,
                                            facets, fn, coeffs,
                                            cstride, perms, get_facet_perm);
  }

  if (M.num_integrals(IntegralType::interior_facet) > 0)
  {
    const std::vector<int> c_offsets = M.coefficient_offsets();
    for (int i : M.integral_ids(IntegralType::interior_facet))
    {
      const auto& fn = M.kernel(IntegralType::interior_facet, i);
      const auto& [coeffs, cstride]
          = coefficients.at({IntegralType::interior_facet, i});
      const std::vector<std::int32_t>& facets = M.interior_facet_domains(i);
      value += impl::assemble_interior_facets(geometry, num_cell_facets, facets,
                                              fn, constants, coeffs, cstride,
                                              c_offsets, perms);
    }
  }

  return value;
}

} // namespace dolfinx::fem::impl
