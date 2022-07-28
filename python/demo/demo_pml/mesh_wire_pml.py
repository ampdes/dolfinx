import sys

import gmsh
from mpi4py import MPI
from numpy import pi


def generate_mesh_wire(radius_wire, l_dom, l_pml, 
                       in_wire_size, on_wire_size, bkg_size, scatt_size, pml_size, 
                       au_tag, bkg_tag, pml_tag, scatt_tag):

    gmsh.initialize(sys.argv)
    if MPI.COMM_WORLD.rank == 0:

        gmsh.model.add("nanowire")

        # A dummy boundary is added for setting a finer mesh
        gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire * 0.8,
                                 angle1=0.0, angle2=2 * pi, tag=1)

        gmsh.model.occ.addCurveLoop([1], tag=1)
        gmsh.model.occ.addPlaneSurface([1], tag=1)

        gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius_wire,
                                 angle1=0, angle2=2 * pi, tag=2)

        gmsh.model.occ.addCurveLoop([2], tag=2)
        gmsh.model.occ.addPlaneSurface([2, 1], tag=2)

        gmsh.model.occ.addCircle(0.0, 0.0, 0.0, 0.8*l_dom/2,
                                 angle1=0, angle2=2 * pi, tag=3)

        gmsh.model.occ.addCurveLoop([3], tag=3)
        gmsh.model.occ.addPlaneSurface([3, 2], tag=3)
        
        gmsh.model.occ.addRectangle(-l_dom / 2, -l_dom / 2, 0, l_dom, l_dom)
        gmsh.model.occ.addRectangle(-l_pml / 2, -l_pml / 2, 0, l_pml, l_pml)

        gmsh.model.occ.remove(dimTags=[(2, 4)], recursive=False)
        gmsh.model.occ.remove(dimTags=[(2, 5)], recursive=False)

        gmsh.model.occ.addPlaneSurface([4, 3], tag=4)
        
        gmsh.model.occ.addPlaneSurface([5, 4], tag=5)
        
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [1, 2], tag=au_tag)
        gmsh.model.addPhysicalGroup(2, [3, 4], tag=bkg_tag)
        gmsh.model.addPhysicalGroup(2, [5], tag=pml_tag)
        gmsh.model.addPhysicalGroup(1, [3], tag=scatt_tag)

        gmsh.model.mesh.setSize([(0, 1)], size=in_wire_size)
        gmsh.model.mesh.setSize([(0, 2)], size=on_wire_size)
        gmsh.model.mesh.setSize([(0, 3)], size=scatt_size)
        gmsh.model.mesh.setSize([(0, x) for x in range(4, 8)], size=bkg_size)
        gmsh.model.mesh.setSize([(0, x) for x in range(8, 12)], size=pml_size)

        gmsh.model.mesh.generate(2)

        return gmsh.model
