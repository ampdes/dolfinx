// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// =====================================
// Matrix-free Conjugate Gradient solver
// =====================================
//
// This demo illustrates how to:
// * Solve a linear partial differential equation using a matrix free CG solver
// * Create and apply Dirichlet boundary conditions
// * Compute errors
//
// .. math::
//    - \nabla^{2} u &= f \quad {\rm in} \ \Omega, \\
//      u &= u_D \quad {\rm on} \ \Gamma_{D}
//
//  Where
// .. math::
//    u_D &= 1 + x^2 + 2y^2, \\
//    f = -6
//
// .. note:: This demo illustrates the use of a matrix-free conjugate
//           gradient solver. Many practical problems will also require
//           a preconditioner to create and efficient solver.

#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>

#include "ghost_layer.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& v)
{
  for (os << "{ "; auto const& e : v)
    os << e << ' ';
  return os << '}';
}

template <typename T>
std::ostream& operator<<(std::ostream& os, std::span<const T> const& v)
{
  for (os << "{ "; auto const& e : v)
    os << e << ' ';
  return os << '}';
}

using namespace dolfinx;

namespace linalg
{
/// Compute vector r = alpha*x + y
/// @param[out] r Result
/// @param[in] alpha
/// @param[in] x
/// @param[in] y
void axpy(auto& r, auto alpha, const auto& x, const auto& y)
{
  std::transform(x.array().begin(), x.array().end(), y.array().begin(),
                 r.mutable_array().begin(),
                 [alpha](auto x, auto y) { return alpha * x + y; });
}

/// Solve problem A.x = b using the Conjugate Gradient method
/// @tparam U The scalar type
/// @tparam ApplyFunction Type of the function object "action"
/// @param[in, out] x Solution vector, may be set to an initial guess
/// @param[in] b RHS Vector
/// @param[in] action Function that provides the action of the linear operator
/// @param[in] kmax Maximum number of iterations
/// @param[in] rtol Relative tolerances for convergence
/// @return The number if iterations
/// @pre It is required that the ghost values of `x` and `b` have been
/// updated before this function is called
template <typename ApplyFunction>
int cg(auto& x, auto& b, ApplyFunction&& action, int kmax = 50,
       double rtol = 1e-8)
{
  using T = typename std::decay_t<decltype(x)>::value_type;

  // Create working vectors
  la::Vector r(b), y(b);

  // Compute initial residual r0 = b - Ax0
  action(x, y);
  axpy(r, T(-1), y, b);

  // Create p work vector
  la::Vector p(r);

  // Iterations of CG
  auto rnorm0 = la::squared_norm(r);
  const auto rtol2 = rtol * rtol;
  auto rnorm = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // Compute y = A p
    action(p, y);

    // Compute alpha = r.r/p.y
    const T alpha = rnorm / la::inner_product(p, y);

    // Update x (x <- x + alpha*p)
    axpy(x, alpha, p, x);

    // Update r (r <- r - alpha*y)
    axpy(r, -alpha, y, r);

    // Update residual norm
    const auto rnorm_new = la::squared_norm(r);
    const T beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p (p <- beta*p + r)
    axpy(p, beta, p, r);
  }

  return k;
}
} // namespace linalg

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  {
    using T = PetscScalar;
    using U = typename dolfinx::scalar_value_type_t<T>;

    MPI_Comm comm = MPI_COMM_WORLD;
    [[maybe_unused]] int rank = dolfinx::MPI::rank(comm);

    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
        comm, {{{0.0, 0.0}, {1.0, 1.0}}}, {10, 10}, mesh::CellType::triangle,
        mesh::create_cell_partitioner(mesh::GhostMode::shared_facet)));

    std::cout << "rank = " << rank << ":"
              << mesh->topology()->index_map(2)->num_ghosts() << std::endl;

    mesh = create_ghost_layer(mesh);

    std::cout << "rank = " << rank << ":"
              << mesh->topology()->index_map(2)->num_ghosts() << std::endl;

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(functionspace_form_poisson_M, "ui", mesh));

    // Prepare and set Constants for the bilinear form
    auto f = std::make_shared<fem::Constant<T>>(-6.0);

    // Define variational forms
    auto L = std::make_shared<fem::Form<T, U>>(
        fem::create_form<T>(*form_poisson_L, {V}, {}, {{"f", f}}, {}));

    // Create two distinct set of cells, deferred and immediate
    std::vector deferred_cells = fem::locate_cells_with_ghost_dofs(*V);

    int tdim = mesh->topology()->dim();
    std::int32_t num_cells = mesh->topology()->index_map(tdim)->size_local();
    std::vector<std::int32_t> all_cells(num_cells);
    std::iota(all_cells.begin(), all_cells.end(), 0);
    std::vector<std::int32_t> non_sharing_cells;

    std::set_difference(
        all_cells.begin(), all_cells.end(), deferred_cells.begin(),
        deferred_cells.end(),
        std::inserter(non_sharing_cells, non_sharing_cells.begin()));

    if (rank == 0)
    {
      std::cout << non_sharing_cells.size() << std::endl;
      std::cout << deferred_cells.size() << std::endl;
    }

    // Action of the bilinear form "a" on a function ui
    auto ui = std::make_shared<fem::Function<T, U>>(V);

    std::map<fem::IntegralType,
             std::vector<std::pair<std::int32_t, std::vector<std::int32_t>>>>
        subdomains;

    subdomains[fem::IntegralType::cell] = {{-1, non_sharing_cells}};
    auto Mi = std::make_shared<fem::Form<T, U>>(fem::create_form<T>(
        *form_poisson_M, {V}, {{"ui", ui}}, {{}}, subdomains));

    subdomains[fem::IntegralType::cell] = {{-1, deferred_cells}};
    auto Md = std::make_shared<fem::Form<T, U>>(fem::create_form<T>(
        *form_poisson_M, {V}, {{"ui", ui}}, {{}}, subdomains));

    // Define boundary condition
    auto u_D = std::make_shared<fem::Function<T, U>>(V);
    u_D->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(1 + x(0, p) * x(0, p) + +2 * x(1, p) * x(1, p));
          return {f, {f.size()}};
        });

    mesh->topology_mutable()->create_connectivity(1, 2);
    const std::vector<std::int32_t> facets
        = mesh::exterior_facet_indices(*mesh->topology());
    std::vector<std::int32_t> bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(u_D, bdofs);

    // Assemble RHS vector
    la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());
    fem::assemble_vector(b.mutable_array(), *L);

    // Apply lifting to account for Dirichlet boundary condition
    // b <- b - A * x_bc
    fem::set_bc<T, U>(ui->x()->mutable_array(), {bc}, T(-1));
    fem::assemble_vector(b.mutable_array(), *Mi);
    fem::assemble_vector(b.mutable_array(), *Md);

    // Communicate ghost values
    b.scatter_rev(std::plus<T>());

    // Set BC dofs to zero (effectively zeroes columns of A)
    fem::set_bc<T, U>(b.mutable_array(), {bc}, T(0));
    b.scatter_fwd();

    // Pack coefficients and constants
    auto coeff_i = fem::allocate_coefficient_storage(*Mi);
    const std::vector<T> constants_i = fem::pack_constants(*Mi);

    auto coeff_d = fem::allocate_coefficient_storage(*Md);
    const std::vector<T> constants_d = fem::pack_constants(*Md);

    // Create function for computing the action of A on x (y = Ax)
    auto action = [&](auto& x, auto& y)
    {
      x.scatter_fwd_begin();

      // Zero y
      y.set(0.0);

      // Update coefficient ui (just copy data from x to ui)
      std::copy(x.array().begin(), x.array().end(),
                ui->x()->mutable_array().begin());

      // Compute action of A on x on subset of cells "i" - local part
      fem::pack_coefficients(*Mi, coeff_i);
      fem::assemble_vector(y.mutable_array(), *Mi,
                           std::span<const T>(constants_i),
                           fem::make_coefficients_span(coeff_i));

      x.scatter_fwd_end();

      // Compute action of A on x on subset of cells "d" - cells with ghost values
      // fem::pack_coefficients(*Md, coeff_d);
      // fem::assemble_vector(y.mutable_array(), *Md,
      //                      std::span<const T>(constants_d),
      //                      fem::make_coefficients_span(coeff_d));

      // Set BC dofs to zero (effectively zeroes rows of A)
      fem::set_bc<T, U>(y.mutable_array(), {bc}, T(0));
    };

    // Compute solution using the conjugate gradient method
    auto u = std::make_shared<fem::Function<T>>(V);
    int num_it = linalg::cg(*u->x(), b, action, 200, 1e-6);

    // Set BC values in the solution vectors
    fem::set_bc<T, U>(u->x()->mutable_array(), {bc}, T(1));

    // Compute L2 error (squared) of the solution vector e = (u - u_d, u
    // - u_d)*dx
    auto E = std::make_shared<fem::Form<T>>(fem::create_form<T, U>(
        *form_poisson_E, {}, {{"uexact", u_D}, {"usol", u}}, {}, {}, mesh));
    T error = fem::assemble_scalar(*E);
    if (dolfinx::MPI::rank(comm) == 0)
    {
      std::cout << "Number of CG iterations " << num_it << std::endl;
      std::cout << "Finite element error (L2 norm (squared)) "
                << std::abs(error) << std::endl;
    }
  }

  MPI_Finalize();

  return 0;
}
