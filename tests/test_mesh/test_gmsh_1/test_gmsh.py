import gmsh
import sys

gmsh.initialize()
gmsh.model.add("veio")

# Usar OpenCASCADE é muito mais simples para sólidos 3D
gmsh.model.occ.addCylinder(0, 0, 0,  # ponto de origem
                            0, 0, 100, # direção e comprimento (100mm)
                            10)         # raio (10mm)

gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)  # malha 3D
gmsh.write("veio.msh")

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()