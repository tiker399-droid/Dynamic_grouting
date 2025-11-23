from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import numpy
from mpi4py import MPI
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 div, dx, ds, grad, inner)

N = 8
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
V = fem.functionspace(domain, ("Lagrange", 1))

uD = fem.Function(V)
x = SpatialCoordinate(domain)
u_ex =  1 + x[0]**2 + 2 * x[1]**2
uD.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
f = -div(grad(u_ex))

u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(domain)
h = 2 * Circumradius(domain)
alpha = fem.Constant(domain, default_scalar_type(10))
a = inner(grad(u), grad(v)) * dx - inner(n, grad(u)) * v * ds
a += - inner(n, grad(v)) * u * ds + alpha / h * inner(u, v) * ds
L = inner(f, v) * dx 
L += - inner(n, grad(v)) * uD * ds + alpha / h * inner(uD, v) * ds

problem = LinearProblem(a, L)
uh = problem.solve()

error_form = fem.form(inner(uh-uD, uh-uD) * dx)
error_local = fem.assemble_scalar(error_form)
errorL2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if domain.comm.rank == 0:
    print(fr"$L^2$-error: {errorL2:.2e}")

error_max = domain.comm.allreduce(numpy.max(numpy.abs(uD.x.array-uh.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max : {error_max:.2e}")

import pyvista

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))# 创建网格
grid.point_data["u"] = uh.x.array.real # 导入数据
grid.set_active_scalars("u") # 设置活动标量
plotter = pyvista.Plotter() # 创建绘图器
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)   # 添加网格
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("nitsche.png")