// Copyright (C) 2010-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "plaza.h"
#include "refine.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>

namespace dolfinx::refinement
{
/// @brief Create a uniformly refined mesh.
///
/// @param[in] mesh Mesh to create a new, refined mesh from
/// @param[in] redistribute If `true` refined mesh is re-partitioned
/// across MPI ranks.
/// @return Refined mesh
template <std::floating_point T>
mesh::Mesh<T> refine(const mesh::Mesh<T>& mesh, bool redistribute = true)
{
  auto topology = mesh.topology();
  assert(topology);

  if (topology->cell_type() != mesh::CellType::interval
      and topology->cell_type() != mesh::CellType::triangle
      and topology->cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  if (topology->cell_type() == mesh::CellType::interval)
  {
    common::Timer t0("PLAZA: refine");
    auto topology = mesh.topology();
    assert(topology);

    if (topology->cell_type() != mesh::CellType::triangle
        and topology->cell_type() != mesh::CellType::tetrahedron)
    {
      throw std::runtime_error("Cell type not supported");
    }

    auto map_e = topology->index_map(1);
    if (!map_e)
      throw std::runtime_error("Edges must be initialised");

    // Get sharing ranks for each edge
    graph::AdjacencyList<int> edge_ranks = map_e->index_to_dest_ranks();

    // Create unique list of ranks that share edges (owners of ghosts
    // plus ranks that ghost owned indices)
    std::vector<int> ranks(edge_ranks.array().begin(),
                           edge_ranks.array().end());
    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

    // Convert edge_ranks from global rank to to neighbourhood ranks
    std::transform(edge_ranks.array().begin(), edge_ranks.array().end(),
                   edge_ranks.array().begin(),
                   [&ranks](auto r)
                   {
                     auto it = std::lower_bound(ranks.begin(), ranks.end(), r);
                     assert(it != ranks.end() and *it == r);
                     return std::distance(ranks.begin(), it);
                   });

    MPI_Comm comm;
    MPI_Dist_graph_create_adjacent(mesh.comm(), ranks.size(), ranks.data(),
                                   MPI_UNWEIGHTED, ranks.size(), ranks.data(),
                                   MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm);

    int tdim = mesh.topology()->dim();
    bool compute_facets = (option == plaza::Option::parent_facet
                           or option == plaza::Option::parent_cell_and_facet);
    bool compute_parent_cell
        = (option == plaza::Option::parent_cell
           or option == plaza::Option::parent_cell_and_facet);

    // Make new vertices in parallel
    const auto [new_vertex_map, new_vertex_coords, xshape]
        = dolfinx::refinement::create_new_vertices(neighbor_comm, shared_edges,
                                                   mesh, marked_edges);

    std::vector<std::int32_t> parent_cell;
    std::vector<std::int8_t> parent_facet;
    std::vector<std::int64_t> indices(num_cell_vertices);

    auto map_c = mesh.topology()->index_map(tdim);
    assert(map_c);
    auto c_to_v = mesh.topology()->connectivity(tdim, 0);

    std::int32_t num_new_vertices_local = std::count(
        marked_edges.begin(),
        marked_edges.begin() + mesh.topology()->index_map(1)->size_local(),
        true);

    std::vector<std::int64_t> global_indices
        = dolfinx::refinement::impl::adjust_indices(
            *mesh.topology()->index_map(0), num_new_vertices_local);

    const int num_cells = map_c->size_local();

    // Iterate over all cells, and refine if cell has a marked edge
    std::vector<std::int64_t> cell_topology;
    for (int c = 0; c < num_cells; ++c)
    {
      // Copy vertices
      auto vertices = c_to_v->links(c);
      for (std::size_t v = 0; v < vertices.size(); ++v)
        indices[v] = global_indices[vertices[v]];

      if (marked_edges[c])
      {
        auto it = new_vertex_map.find(c);
        assert(it != new_vertex_map.end());
      }
      else
      {
        // Copy over existing cell to new topology
        for (auto v : vertices)
          cell_topology.push_back(global_indices[v]);

        if (compute_parent_cell)
          parent_cell.push_back(c);

        if (compute_facets)
        {
          parent_facet.insert(parent_facet.end(), {0});
        }
      }
      else
      {
        // FIXME: this has an expensive dynamic memory allocation
        simplex_set = get_simplices(indices, longest_edge, tdim, uniform);

        if (compute_parent_cell)
        {
          for (std::int32_t i = 0; i < 2; ++i)
            parent_cell.push_back(c);
        }

        if (compute_facets)
        {
          std::vector<std::int8_t> npf;
          if (tdim == 3)
            npf = compute_parent_facets<3>(simplex_set);
          else
            npf = compute_parent_facets<2>(simplex_set);
          parent_facet.insert(parent_facet.end(), npf.begin(), npf.end());
        }

        // Convert from cell local index to mesh index and add to cells
        for (std::int32_t v : simplex_set)
          cell_topology.push_back(indices[v]);
      }
    }

    assert(cell_topology.size() % num_cell_vertices == 0);
    std::vector<std::int32_t> offsets(
        cell_topology.size() / num_cell_vertices + 1, 0);
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
      offsets[i + 1] = offsets[i] + num_cell_vertices;
    graph::AdjacencyList cell_adj(std::move(cell_topology), std::move(offsets));

    return {std::move(cell_adj), std::move(new_vertex_coords), xshape,
            std::move(parent_cell), std::move(parent_facet)};

    //   // Report the number of refined cells
    //   const int D = topology->dim();
    //   const std::int64_t n0 = topology->index_map(D)->size_global();
    //   const std::int64_t n1
    //       = refined_mesh.topology()->index_map(D)->size_global();
    //   LOG(INFO) << "Number of cells increased from " << n0 << " to " << n1 <<
    //   "
    //   ("
    //             << 100.0
    //                    * (static_cast<double>(n1) / static_cast<double>(n0)
    //                    - 1.0)
    //             << "%% increase).";

    //   return refined_mesh;
  }
  // else
  // {
  //
  auto [refined_mesh, parent_cell, parent_facet]
      = plaza::refine(mesh, redistribute, plaza::Option::none);

  // Report the number of refined cells
  const int D = topology->dim();
  const std::int64_t n0 = topology->index_map(D)->size_global();
  const std::int64_t n1 = refined_mesh.topology()->index_map(D)->size_global();
  LOG(INFO) << "Number of cells increased from " << n0 << " to " << n1 << " ("
            << 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0)
            << "%% increase).";

  return refined_mesh;
  //}
}

/// @brief Create a locally refined mesh.
///
/// @param[in] mesh Mesh to create a new, refined mesh from.
/// @param[in] edges Indices of the edges that should be split during
/// refinement. mesh::compute_incident_entities can be used to compute
/// the edges that are incident to other entities, e.g. incident to
/// cells.
/// @param[in] redistribute If `true` refined mesh is re-partitioned
/// across MPI ranks.
/// @return Refined mesh.
template <std::floating_point T>
mesh::Mesh<T> refine(const mesh::Mesh<T>& mesh,
                     std::span<const std::int32_t> edges,
                     bool redistribute = true)
{
  auto topology = mesh.topology();
  assert(topology);
  if (topology->cell_type() != mesh::CellType::interval
      and topology->cell_type() != mesh::CellType::triangle
      and topology->cell_type() != mesh::CellType::tetrahedron)
  {
    throw std::runtime_error("Refinement only defined for simplices");
  }

  auto [refined_mesh, parent_cell, parent_facet]
      = plaza::refine(mesh, edges, redistribute, plaza::Option::none);

  // Report the number of refined cells
  const int D = topology->dim();
  const std::int64_t n0 = topology->index_map(D)->size_global();
  const std::int64_t n1 = refined_mesh.topology()->index_map(D)->size_global();
  LOG(INFO) << "Number of cells increased from " << n0 << " to " << n1 << " ("
            << 100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0)
            << "%% increase).";

  return refined_mesh;
}

} // namespace dolfinx::refinement
