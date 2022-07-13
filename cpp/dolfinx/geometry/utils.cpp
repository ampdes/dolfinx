// Copyright (C) 2006-2021 Chris N. Richardson, Anders Logg, Garth N. Wells and
// Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include "BoundingBoxTree.h"
#include "gjk.h"
#include <deque>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------
// Check whether bounding box is a leaf node
constexpr bool is_leaf(const std::array<int, 2>& bbox)
{
  // Leaf nodes are marked by setting child_0 equal to child_1
  return bbox[0] == bbox[1];
}
//-----------------------------------------------------------------------------
/// A point `x` is inside a bounding box `b` if each component of its
/// coordinates lies within the range `[b(0,i), b(1,i)]` that defines the bounds
/// of the bounding box, b(0,i) <= x[i] <= b(1,i) for i = 0, 1, 2
bool point_in_bbox(const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b,
                   const xt::xtensor_fixed<double, xt::xshape<3>>& x)
{
  constexpr double rtol = 1e-14;
  double eps;
  bool in = true;
  for (int i = 0; i < 3; i++)
  {
    eps = rtol * (b(1, i) - b(0, i));
    in &= x[i] >= (b(0, i) - eps);
    in &= x[i] <= (b(1, i) + eps);
  }

  return in;
}
//-----------------------------------------------------------------------------
/// A bounding box "a" is contained inside another bounding box "b", if each
/// of its intervals [a(0,i), a(1,i)] is contained in [b(0,i), b(1,i)],
/// a(0,i) <= b(1, i) and a(1,i) >= b(0, i)
bool bbox_in_bbox(const xt::xtensor_fixed<double, xt::xshape<2, 3>>& a,
                  const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b)
{
  constexpr double rtol = 1e-14;
  double eps;
  bool in = true;

  for (int i = 0; i < 3; i++)
  {
    eps = rtol * (b(1, i) - b(0, i));
    in &= a(1, i) >= (b(0, i) - eps);
    in &= a(0, i) <= (b(1, i) + eps);
  }

  return in;
}
//-----------------------------------------------------------------------------
// Compute closest entity {closest_entity, R2} (recursive)
std::pair<std::int32_t, double>
_compute_closest_entity(const geometry::BoundingBoxTree& tree,
                        const xt::xtensor_fixed<double, xt::xshape<3>>& point,
                        int node, const mesh::Mesh& mesh,
                        std::int32_t closest_entity, double R2)
{
  // Get children of current bounding box node (child_1 denotes entity
  // index for leaves)
  const std::array bbox = tree.bbox(node);
  double r2;
  if (is_leaf(bbox))
  {
    // If point cloud tree the exact distance is easy to compute
    if (tree.tdim() == 0)
    {
      xt::xtensor_fixed<double, xt::xshape<3>> diff
          = xt::row(tree.get_bbox(node), 0);
      diff -= point;
      r2 = xt::norm_sq(diff)();
    }
    else
    {
      r2 = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
      // If bounding box closer than previous closest entity, use gjk to
      // obtain exact distance to the convex hull of the entity
      if (r2 <= R2)
      {
        r2 = geometry::squared_distance(mesh, tree.tdim(),
                                        xtl::span(&bbox[1], 1),
                                        xt::reshape_view(point, {1, 3}))[0];
      }
    }

    // If entity is closer than best result so far, return it
    if (r2 <= R2)
    {
      closest_entity = bbox[1];
      R2 = r2;
    }

    return {closest_entity, R2};
  }
  else
  {
    // If bounding box is outside radius, then don't search further
    r2 = geometry::compute_squared_distance_bbox(tree.get_bbox(node), point);
    if (r2 > R2)
      return {closest_entity, R2};

    // Check both children
    // We use R2 (as opposed to r2), as a bounding box can be closer
    // than the actual entity
    std::pair<int, double> p0 = _compute_closest_entity(
        tree, point, bbox[0], mesh, closest_entity, R2);
    std::pair<int, double> p1 = _compute_closest_entity(
        tree, point, bbox[1], mesh, p0.first, p0.second);
    return p1;
  }
}
//-----------------------------------------------------------------------------
/// Compute collisions with a single point
/// @param[in] tree The bounding box tree
/// @param[in] points The points (shape=(num_points, 3))
/// @param[in, out] entities The list of colliding entities (local to process)
void _compute_collisions_point(
    const geometry::BoundingBoxTree& tree,
    const xt::xtensor_fixed<double, xt::xshape<3>>& p,
    std::vector<int>& entities)
{
  std::deque<std::int32_t> stack;
  int next = tree.num_bboxes() - 1;

  while (next != -1)
  {
    std::array bbox = tree.bbox(next);
    next = -1;

    if (is_leaf(bbox))
    {
      // If box is a leaf node then add it to the list of colliding entities
      entities.push_back(bbox[1]);
    }
    else
    {
      // Check whether the point collides with child nodes (left and right)
      bool left = point_in_bbox(tree.get_bbox(bbox[0]), p);
      bool right = point_in_bbox(tree.get_bbox(bbox[1]), p);
      if (left && right)
      {
        // If the point collides with both child nodes, add the right node to
        // the stack (for later visiting) and continue the tree traversal with
        // the left subtree
        stack.push_back(bbox[1]);
        next = bbox[0];
      }
      else if (left)
      {
        // Traverse the current node's left subtree
        next = bbox[0];
      }
      else if (right)
      {
        // Traverse the current node's right subtree
        next = bbox[1];
      }
    }
    // If tree traversal reaches a dead end (box is a leaf node or no collision
    // detected), check the stack for deferred subtrees.
    if (next == -1 and !stack.empty())
    {
      next = stack.back();
      stack.pop_back();
    }
  }
}
//-----------------------------------------------------------------------------
// Compute collisions with tree (recursive)
void _compute_collisions_tree(const geometry::BoundingBoxTree& A,
                              const geometry::BoundingBoxTree& B, int node_A,
                              int node_B,
                              std::vector<std::array<int, 2>>& entities)
{
  // If bounding boxes don't collide, then don't search further
  if (!bbox_in_bbox(A.get_bbox(node_A), B.get_bbox(node_B)))
    return;

  // Get bounding boxes for current nodes
  const std::array bbox_A = A.bbox(node_A);
  const std::array bbox_B = B.bbox(node_B);

  // Check whether we've reached a leaf in A or B
  const bool is_leaf_A = is_leaf(bbox_A);
  const bool is_leaf_B = is_leaf(bbox_B);
  if (is_leaf_A and is_leaf_B)
  {
    // If both boxes are leaves (which we know collide), then add them
    // child_1 denotes entity for leaves
    entities.push_back({bbox_A[1], bbox_B[1]});
  }
  else if (is_leaf_A)
  {
    // If we reached the leaf in A, then descend B
    _compute_collisions_tree(A, B, node_A, bbox_B[0], entities);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], entities);
  }
  else if (is_leaf_B)
  {
    // If we reached the leaf in B, then descend A
    _compute_collisions_tree(A, B, bbox_A[0], node_B, entities);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, entities);
  }
  else if (node_A > node_B)
  {
    // At this point, we know neither is a leaf so descend the largest
    // tree first. Note that nodes are added in reverse order with the
    // top bounding box at the end so the largest tree (the one with the
    // the most boxes left to traverse) has the largest node number.
    _compute_collisions_tree(A, B, bbox_A[0], node_B, entities);
    _compute_collisions_tree(A, B, bbox_A[1], node_B, entities);
  }
  else
  {
    _compute_collisions_tree(A, B, node_A, bbox_B[0], entities);
    _compute_collisions_tree(A, B, node_A, bbox_B[1], entities);
  }

  // Note that cases above can be collected in fewer cases but this way
  // the logic is easier to follow.
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
geometry::BoundingBoxTree
geometry::create_midpoint_tree(const mesh::Mesh& mesh, int tdim,
                               const xtl::span<const std::int32_t>& entities)
{
  LOG(INFO) << "Building point search tree to accelerate distance queries for "
               "a given topological dimension and subset of entities.";

  const std::vector<double> midpoints
      = mesh::compute_midpoints(mesh, tdim, entities);
  std::vector<std::pair<std::array<double, 3>, std::int32_t>> points(
      entities.size());
  for (std::size_t i = 0; i < points.size(); ++i)
  {
    for (std::size_t j = 0; j < 3; ++j)
      points[i].first[j] = midpoints[3 * i + j];
    points[i].second = entities[i];
  }

  // Build tree
  return geometry::BoundingBoxTree(points);
}
//-----------------------------------------------------------------------------
std::vector<std::array<int, 2>>
geometry::compute_collisions(const BoundingBoxTree& tree0,
                             const BoundingBoxTree& tree1)
{
  // Call recursive find function
  std::vector<std::array<int, 2>> entities;
  if (tree0.num_bboxes() > 0 and tree1.num_bboxes() > 0)
  {
    _compute_collisions_tree(tree0, tree1, tree0.num_bboxes() - 1,
                             tree1.num_bboxes() - 1, entities);
  }

  return entities;
}
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
geometry::compute_collisions(const BoundingBoxTree& tree,
                             const xt::xtensor<double, 2>& points)
{
  if (tree.num_bboxes() > 0)
  {
    std::vector<std::int32_t> entities, offsets(points.shape(0) + 1, 0);
    entities.reserve(points.shape(0));
    for (std::size_t p = 0; p < points.shape(0); ++p)
    {
      _compute_collisions_point(tree, xt::row(points, p), entities);
      offsets[p + 1] = entities.size();
    }

    return graph::AdjacencyList<std::int32_t>(std::move(entities),
                                              std::move(offsets));
  }
  else
  {
    return graph::AdjacencyList<std::int32_t>(
        std::vector<std::int32_t>(),
        std::vector<std::int32_t>(points.shape(0) + 1, 0));
  }
}
//-----------------------------------------------------------------------------
std::vector<std::int32_t> geometry::compute_closest_entity(
    const BoundingBoxTree& tree, const BoundingBoxTree& midpoint_tree,
    const mesh::Mesh& mesh, const xt::xtensor<double, 2>& points)
{
  assert(points.shape(1) == 3);
  if (tree.num_bboxes() == 0)
    return std::vector<std::int32_t>(points.shape(0), -1);
  else
  {
    double R2;
    double initial_entity;
    std::array<int, 2> leaves;
    std::vector<std::int32_t> entities;
    entities.reserve(points.shape(0));
    for (std::size_t i = 0; i < points.shape(0); i++)
    {
      // Use midpoint tree to find initial closest entity to the point.
      // Start by using a leaf node as the initial guess for the input
      // entity
      leaves = midpoint_tree.bbox(0);
      assert(is_leaf(leaves));
      initial_entity = leaves[0];
      xt::xtensor_fixed<double, xt::xshape<3>> diff
          = xt::row(midpoint_tree.get_bbox(0), 0);
      diff -= xt::row(points, i);
      R2 = xt::norm_sq(diff)();

      // Use a recursive search through the bounding box tree
      // to find determine the entity with the closest midpoint.
      // As the midpoint tree only consist of points, the distance
      // queries are lightweight.
      const auto [m_index, m_distance2] = _compute_closest_entity(
          midpoint_tree, xt::reshape_view(xt::row(points, i), {1, 3}),
          midpoint_tree.num_bboxes() - 1, mesh, initial_entity, R2);

      // Use a recursive search through the bounding box tree to
      // determine which entity is actually closest.
      // Uses the entity with the closest midpoint as initial guess, and
      // the distance from the midpoint to the point of interest as the
      // initial search radius.
      const auto [index, distance2] = _compute_closest_entity(
          tree, xt::reshape_view(xt::row(points, i), {1, 3}),
          tree.num_bboxes() - 1, mesh, m_index, m_distance2);

      entities.push_back(index);
    }

    return entities;
  }
}

//-----------------------------------------------------------------------------
double geometry::compute_squared_distance_bbox(
    const xt::xtensor_fixed<double, xt::xshape<2, 3>>& b,
    const xt::xtensor_fixed<double, xt::xshape<3>>& x)
{
  const xt::xtensor_fixed<double, xt::xshape<3>> d0 = x - xt::row(b, 0);
  const xt::xtensor_fixed<double, xt::xshape<3>> d1 = x - xt::row(b, 1);
  auto _d0 = xt::where(d0 > 0.0, 0, d0);
  auto _d1 = xt::where(d1 < 0.0, 0, d1);
  return xt::norm_sq(_d0)() + xt::norm_sq(_d1)();
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
geometry::shortest_vector(const mesh::Mesh& mesh, int dim,
                          const xtl::span<const std::int32_t>& entities,
                          const xt::xtensor<double, 2>& points)
{
  assert(points.shape(1) == 3);
  const int tdim = mesh.topology().dim();
  const mesh::Geometry& geometry = mesh.geometry();
  xtl::span<const double> geom_dofs = geometry.x();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
  xt::xtensor<double, 2> shortest_vectors({entities.size(), 3});
  if (dim == tdim)
  {
    for (std::size_t e = 0; e < entities.size(); e++)
    {
      auto dofs = x_dofmap.links(entities[e]);
      xt::xtensor<double, 2> nodes({dofs.size(), 3});
      for (std::size_t i = 0; i < dofs.size(); ++i)
      {
        const int pos = 3 * dofs[i];
        for (std::size_t j = 0; j < 3; ++j)
          nodes(i, j) = geom_dofs[pos + j];
      }

      xt::row(shortest_vectors, e) = geometry::compute_distance_gjk(
          xt::reshape_view(xt::row(points, e), {1, 3}), nodes);
    }
  }
  else
  {
    mesh.topology_mutable().create_connectivity(dim, tdim);
    mesh.topology_mutable().create_connectivity(tdim, dim);
    auto e_to_c = mesh.topology().connectivity(dim, tdim);
    assert(e_to_c);
    auto c_to_e = mesh.topology_mutable().connectivity(tdim, dim);
    assert(c_to_e);
    for (std::size_t e = 0; e < entities.size(); e++)
    {
      const std::int32_t index = entities[e];

      // Find attached cell
      assert(e_to_c->num_links(index) > 0);
      const std::int32_t c = e_to_c->links(index)[0];

      // Find local number of entity wrt cell
      auto cell_entities = c_to_e->links(c);
      auto it0 = std::find(cell_entities.begin(), cell_entities.end(), index);
      assert(it0 != cell_entities.end());
      const int local_cell_entity = std::distance(cell_entities.begin(), it0);

      // Tabulate geometry dofs for the entity
      auto dofs = x_dofmap.links(c);
      const std::vector<int> entity_dofs
          = geometry.cmap().create_dof_layout().entity_closure_dofs(
              dim, local_cell_entity);
      xt::xtensor<double, 2> nodes({entity_dofs.size(), 3});
      for (std::size_t i = 0; i < entity_dofs.size(); i++)
      {
        const int pos = 3 * dofs[entity_dofs[i]];
        for (std::size_t j = 0; j < 3; ++j)
          nodes(i, j) = geom_dofs[pos + j];
      }

      xt::row(shortest_vectors, e) = compute_distance_gjk(
          xt::reshape_view(xt::row(points, e), {1, 3}), nodes);
    }
  }

  return shortest_vectors;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 1>
geometry::squared_distance(const mesh::Mesh& mesh, int dim,
                           const xtl::span<const std::int32_t>& entities,
                           const xt::xtensor<double, 2>& points)
{
  return xt::norm_sq(shortest_vector(mesh, dim, entities, points), {1});
}
//-------------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t> geometry::compute_colliding_cells(
    const mesh::Mesh& mesh,
    const graph::AdjacencyList<std::int32_t>& candidate_cells,
    const xt::xtensor<double, 2>& points)
{
  std::vector<std::int32_t> offsets = {0};
  offsets.reserve(candidate_cells.num_nodes() + 1);
  std::vector<std::int32_t> colliding_cells;
  constexpr double eps2 = 1e-20;
  const int tdim = mesh.topology().dim();
  for (std::int32_t i = 0; i < candidate_cells.num_nodes(); i++)
  {
    auto cells = candidate_cells.links(i);
    xt::xtensor<double, 2> _point({cells.size(), 3});
    for (std::size_t j = 0; j < cells.size(); j++)
      xt::row(_point, j) = xt::row(points, i);

    xt::xtensor<double, 1> distances_sq
        = geometry::squared_distance(mesh, tdim, cells, _point);
    for (std::size_t j = 0; j < cells.size(); j++)
      if (distances_sq[j] < eps2)
        colliding_cells.push_back(cells[j]);

    offsets.push_back(colliding_cells.size());
  }

  return graph::AdjacencyList<std::int32_t>(std::move(colliding_cells),
                                            std::move(offsets));
}
//-------------------------------------------------------------------------------
int geometry::compute_first_colliding_cell(
    const mesh::Mesh& mesh, const geometry::BoundingBoxTree& tree,
    const xt::xtensor_fixed<double, xt::xshape<3>>& point)
{
  // Compute colliding bounding boxes(cell candidates)
  std::vector<std::int32_t> cell_candidates;
  _compute_collisions_point(tree, point, cell_candidates);

  if (cell_candidates.empty())
    return -1;
  else
  {
    constexpr double eps2 = 1e-20;
    const mesh::Geometry& geometry = mesh.geometry();
    xtl::span<const double> geom_dofs = geometry.x();
    const graph::AdjacencyList<std::int32_t>& x_dofmap = geometry.dofmap();
    const std::size_t num_nodes = geometry.cmap().dim();
    xt::xtensor<double, 2> coordinate_dofs({num_nodes, std::size_t(3)});
    for (auto cell : cell_candidates)
    {
      auto dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < num_nodes; ++i)
        common::impl::copy_N<3>(std::next(geom_dofs.begin(), 3 * dofs[i]),
                                std::next(coordinate_dofs.begin(), 3 * i));
      // Fix
      xt::xtensor_fixed<double, xt::xshape<3>> shortest_vector
          = geometry::compute_distance_gjk(xt::reshape_view(point, {1, 3}),
                                           coordinate_dofs);
      double norm = 0;
      std::for_each(shortest_vector.cbegin(), shortest_vector.cend(),
                    [&norm](const double e) { norm += std::pow(e, 2); });

      if (norm < eps2)
        return cell;
    }
  }
  return -1;
}

//-------------------------------------------------------------------------------
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>,
           std::vector<double>, std::vector<std::int32_t>>
geometry::determine_point_ownership(const mesh::Mesh& mesh,
                                    const xt::xtensor<double, 2>& points)
{
  const MPI_Comm& comm = mesh.comm();

  // Create a global bounding-box tree to find candidate processes with cells
  // that could collide with thte points
  constexpr double padding = 0.0001;
  const int tdim = mesh.topology().dim();
  const auto cell_map = mesh.topology().index_map(tdim);
  const int num_cells = cell_map->size_local();
  // NOTE: Should we send the cells in as input?
  std::vector<std::int32_t> cells(num_cells, 0);
  std::iota(cells.begin(), cells.end(), 0);
  dolfinx::geometry::BoundingBoxTree bb(mesh, tdim, cells, padding);
  dolfinx::geometry::BoundingBoxTree global_bbtree
      = bb.create_global_tree(comm);

  // Compute collisions:
  // For each point in `x` get the processes it should be sent to
  dolfinx::graph::AdjacencyList<std::int32_t> collisions
      = dolfinx::geometry::compute_collisions(global_bbtree, points);

  // Get unique list of outgoing ranks
  std::vector<std::int32_t> out_ranks = collisions.array();
  std::sort(out_ranks.begin(), out_ranks.end());
  out_ranks.erase(std::unique(out_ranks.begin(), out_ranks.end()),
                  out_ranks.end());

  // Compute incoming edges (source processes)
  std::vector<int> in_ranks
      = dolfinx::MPI::compute_graph_edges_nbx(comm, out_ranks);
  std::sort(in_ranks.begin(), in_ranks.end());

  // Create neighborhood communicator in forward direction
  MPI_Comm forward_comm;
  MPI_Dist_graph_create_adjacent(
      comm, in_ranks.size(), in_ranks.data(), MPI_UNWEIGHTED, out_ranks.size(),
      out_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &forward_comm);

  // Compute map from global mpi rank to neighbor rank, "collisions" uses
  // global rank
  std::map<std::int32_t, std::int32_t> rank_to_neighbor;
  for (std::size_t i = 0; i < out_ranks.size(); i++)
    rank_to_neighbor[out_ranks[i]] = i;

  // Count the number of points to send per neighbor process
  std::vector<std::int32_t> send_sizes(out_ranks.size());
  for (std::size_t i = 0; i < points.shape(0); ++i)
    for (const auto& p : collisions.links(i))
      send_sizes[rank_to_neighbor[p]] += 3;

  // Compute receive sizes
  std::vector<std::int32_t> recv_sizes(in_ranks.size());
  send_sizes.reserve(1);
  recv_sizes.reserve(1);
  MPI_Request sizes_request;
  MPI_Ineighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                         MPI_INT, forward_comm, &sizes_request);

  // Compute sending offsets
  std::vector<std::int32_t> send_offsets(send_sizes.size() + 1, 0);
  std::partial_sum(send_sizes.begin(), send_sizes.end(),
                   std::next(send_offsets.begin(), 1));

  // Pack data to send and store unpack map
  std::vector<double> send_data(send_offsets.back());
  std::vector<std::int32_t> counter(send_sizes.size(), 0);
  // unpack map: [index in adj list][pos in x]
  std::vector<std::int32_t> unpack_map(send_offsets.back() / 3);
  for (std::size_t i = 0; i < points.shape(0); ++i)
  {
    const auto point = xt::row(points, i);
    for (const auto& p : collisions.links(i))
    {
      int neighbor = rank_to_neighbor[p];
      int pos = send_offsets[neighbor] + counter[neighbor];
      auto it = std::next(send_data.begin(), pos);
      std::copy(point.begin(), point.end(), it);
      unpack_map[pos / 3] = i;
      counter[neighbor] += 3;
    }
  }

  MPI_Wait(&sizes_request, MPI_STATUS_IGNORE);
  std::vector<std::int32_t> recv_offsets(in_ranks.size() + 1, 0);
  std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                   std::next(recv_offsets.begin(), 1));

  xt::xtensor<double, 2> received_points(
      {std::size_t(recv_offsets.back() / 3), 3});
  MPI_Neighbor_alltoallv(send_data.data(), send_sizes.data(),
                         send_offsets.data(), MPI_DOUBLE,
                         received_points.data(), recv_sizes.data(),
                         recv_offsets.data(), MPI_DOUBLE, forward_comm);

  // Each process checks which points collides with a cell on the process
  const int rank = dolfinx::MPI::rank(comm);
  std::vector<std::int32_t> cell_indicator(received_points.shape(0));
  std::vector<std::int32_t> colliding_cells(received_points.shape(0));
  for (std::size_t p = 0; p < received_points.shape(0); ++p)
  {
    const int colliding_cell = geometry::compute_first_colliding_cell(
        mesh, bb, xt::row(received_points, p));
    cell_indicator[p] = (colliding_cell >= 0) ? rank : -1;
    colliding_cells[p] = colliding_cell;
  }
  // Create neighborhood communicator in the reverse direction: send back col to
  // requesting processes
  MPI_Comm reverse_comm;
  MPI_Dist_graph_create_adjacent(
      comm, out_ranks.size(), out_ranks.data(), MPI_UNWEIGHTED, in_ranks.size(),
      in_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &reverse_comm);

  // Reuse sizes and offsets from first communication set
  // but divide by three
  {
    auto rescale = [](auto& x)
    {
      std::transform(x.cbegin(), x.cend(), x.begin(),
                     [](auto e) { return (e / 3); });
    };
    rescale(recv_sizes);
    rescale(recv_offsets);
    rescale(send_sizes);
    rescale(send_offsets);

    // The communication is reversed, so swap recv to send offsets
    std::swap(recv_sizes, send_sizes);
    std::swap(recv_offsets, send_offsets);
  }

  std::vector<std::int32_t> recv_ranks(recv_offsets.back());
  MPI_Neighbor_alltoallv(
      cell_indicator.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), recv_ranks.data(),
      recv_sizes.data(), recv_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), reverse_comm);

  std::vector<std::int32_t> point_owners(points.shape(0), -1);
  for (std::size_t i = 0; i < unpack_map.size(); i++)
  {
    const std::int32_t pos = unpack_map[i];
    // Only insert new owner if no owner has previously been found
    if ((recv_ranks[i] >= 0) && (point_owners[pos] == -1))
      point_owners[pos] = recv_ranks[i];
  }
  // Communication is reversed again to send dest ranks to all processes
  std::swap(send_sizes, recv_sizes);
  std::swap(send_offsets, recv_offsets);

  // Pack ownership data
  std::vector<std::int32_t> send_owners(send_offsets.back());
  std::fill(counter.begin(), counter.end(), 0);
  for (std::size_t i = 0; i < points.shape(0); ++i)
  {
    for (const auto& p : collisions.links(i))
    {
      int neighbor = rank_to_neighbor[p];
      send_owners[send_offsets[neighbor] + counter[neighbor]++]
          = point_owners[i];
    }
  }

  // Send ownership info
  std::vector<std::int32_t> dest_ranks(recv_offsets.back());
  MPI_Neighbor_alltoallv(
      send_owners.data(), send_sizes.data(), send_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), dest_ranks.data(),
      recv_sizes.data(), recv_offsets.data(),
      dolfinx::MPI::mpi_type<std::int32_t>(), forward_comm);

  // Unpack dest ranks if point owner is this rank
  std::vector<std::int32_t> owned_recv_ranks;
  owned_recv_ranks.reserve(recv_offsets.back());
  std::vector<double> owned_recv_points;
  std::vector<std::int32_t> owned_recv_cells;
  for (std::size_t i = 0; i < in_ranks.size(); i++)
  {
    for (std::int32_t j = recv_offsets[i]; j < recv_offsets[i + 1]; j++)
    {
      if (rank == dest_ranks[j])
      {
        owned_recv_ranks.push_back(in_ranks[i]);
        auto point = xt::row(received_points, j);
        owned_recv_points.insert(owned_recv_points.end(), point.cbegin(),
                                 point.cend());
        owned_recv_cells.push_back(colliding_cells[j]);
      }
    }
  }

  MPI_Comm_free(&forward_comm);
  MPI_Comm_free(&reverse_comm);

  return std::make_tuple(point_owners, owned_recv_ranks, owned_recv_points,
                         owned_recv_cells);
};