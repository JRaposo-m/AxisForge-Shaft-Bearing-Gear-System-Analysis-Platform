# axisforge.solvers.numerics

Generic numerical methods shared by the AxisForge solvers and by design-studies studies. The
package knows nothing about shafts, bearings or gears. It contains one solver, `newton`, a
Newton-Raphson method for a nonlinear system `F(x) = 0` with `x` in R^n (a scalar problem is
`n = 1`).

## Contents

```
numerics/
    __init__.py      eager re-export of newton, NewtonOptions, NewtonError
    newton.py        newton, NewtonOptions, NewtonError
    tests/           test_newton.py
```

The result containers `NewtonResult` and `IterationRecord` are in
`axisforge/results/numerics/newton_raphson/`.

## Entry points

| Task | Call | Returns |
|---|---|---|
| Solve `F(x) = 0` | `newton(residual, x0, *, jacobian=None, options=None, callback=None, name="", label="")` | `NewtonResult` |
| Choose the method | `NewtonOptions(...)` (frozen; checked at construction) | options |
| Summary and history rows | `NewtonResult.record()`, `NewtonResult.records()` | `dict`, `list[dict]` |

```python
import numpy as np
from axisforge.solvers.numerics import NewtonOptions, newton

# 1. Scalar, finite-difference Jacobian, default options
r = newton(lambda x: x**2 - 2.0, [1.0], name="sqrt2", label="demo")

# 2. System with an analytic Jacobian (circle and parabola)
F = lambda v: np.array([v[0]**2 + v[1]**2 - 4.0, v[1] - v[0]**2 + 1.0])
J = lambda v: np.array([[2*v[0], 2*v[1]], [-2*v[0], 1.0]])
r = newton(F, [1.0, 1.0], jacobian=J, name="circle_parabola", label="analytic J")

# 3. Finite-difference Jacobian, line search, mixed units: displacement u [mm] and rotation
#    th [rad] of a rigid body on two hardening springs; residuals force [N], moment [N mm]
def residual(x):
    u, th = x
    d1, d2 = u - 50.0 * th, u + 50.0 * th
    q1, q2 = 1e4 * d1 + 1e7 * d1**3, 1e4 * d2 + 1e7 * d2**3
    return np.array([q1 + q2 - 2000.0, 50.0 * (q2 - q1) - 1.0e4])

opts = NewtonOptions(globalisation="line_search", jacobian_fd="central",
                     x_scale=[1e-2, 1e-4], f_scale=[1e3, 1e5])
r = newton(residual, [0.0, 0.0], options=opts, name="rigid_body", label="F 2 kN, M 10 N m")
print(r.converged, r.reason, r.n_iter)
```

## Conventions

* `x`, `F` and the Jacobian are in the units of the caller. Every norm (stopping tests,
  `max_step`, history) is taken of the scaled vectors `x / x_scale`, `dx / x_scale` and
  `F / f_scale`; set the scales when the unknowns or the residuals have mixed units.
* A solve that did not converge is returned with `converged=False` and its `reason`.
* `history[0]` is the initial point; `n_iter` counts steps taken.

## Standards

None. Method sources (not in the library): Dennis and Schnabel (1996), Nocedal and Wright
(2006), Kelley (2003); see the module docstring.
