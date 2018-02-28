// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dolfin
{

class Mesh;

namespace function
{
class Function;
}

namespace io
{

/// Write VTK Mesh representation

class VTKWriter
{
public:
  /// Mesh writer
  static void write_mesh(const Mesh& mesh, std::size_t cell_dim,
                         std::string file);

  /// Cell data writer
  static void write_cell_data(const function::Function& u, std::string file);

private:
  // Write cell data (ascii)
  static std::string ascii_cell_data(const Mesh& mesh,
                                     const std::vector<std::size_t>& offset,
                                     const std::vector<double>& values,
                                     std::size_t dim, std::size_t rank);

  // Mesh writer (ascii)
  static void write_ascii_mesh(const Mesh& mesh, std::size_t cell_dim,
                               std::string file);

  // Get VTK cell type
  static std::uint8_t vtk_cell_type(const Mesh& mesh, std::size_t cell_dim);
};
}
}