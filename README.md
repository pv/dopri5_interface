dopri5_interface.h
==================

DOPRI5 interface for modern C++

This is a simple C++ interface for the DOPRI5 solver for Ordinary
Differential Equations by Hairer & Wanner.  The DOPRI5 code is written
in Fortran, and is a well-known implementation of an explicit adaptive
Runge-Kutta method.  The [dopri5_interface.h](@ref dopri5_interface.h) file contains a C++
wrapper interface for this solver.

Features:

- Support integration of equations with different vector types:
  `double`, `std::complex`, ``double[]``, ``std::complex[]``, or
  [Eigen](https://eigen.tuxfamily.org) matrices/vectors.

- Output into a container, or using a dense output callback.

- When using lambdas or functors as callbacks, the compiler has enough
  information to inline everything into a single callback routine for
  DOPRI5, so that the wrapper incurs zero extra overhead compared to
  writing everything manually.

Contents
--------

- [dopri5::solve(x0, xend, func, solout)](@ref dopri5::solve)
- [dopri5::solve_at(xbegin, xend, ybegin, func)](@ref dopri5::solve_at)
- [dopri5::solve_at(xbegin, xend, xbreak_begin, xbreak_end, ybegin, func)](@ref dopri5::solve_at)
- [dopri5::dense_solution](@ref dopri5::dense_solution),
  [dopri5::dense_solution::operator()](@ref dopri5::dense_solution::operator() ),
  [dopri5::dense_solution::get](@ref dopri5::dense_solution::get),
- [dopri5::success_status](@ref dopri5::success_status)

Examples
--------

A simple example, with output into a container:

```cpp
    // Solve equations dy0/dx = y0,  dy1/dx = -y1
    std::vector<double> xs({0, 0.25, 0.5, 0.75, 1});
    std::vector<double[2]> ys(xs.size());
    std::vector<double[2]>::iterator yend;

    auto func = [](double x, const auto& y, auto& dy) {
        dy[0] = y[0];
        dy[1] = -y[1];
    };

    double y0[2] = {1, 2};
    yend = dopri5::solve_at(xs.begin(), xs.end(), y0, func, ys.begin());

    for (int i = 0; i < yend - ys.begin(); ++i) {
        std::cout << "x[" << i << "] = " << xs[i]
                  << "; y[" << i << "] = {"
                  << ys[i][0] << ", " << ys[i][1]
                  << "}" << std::endl;
    }
```

Another example, using a dense output callback and some matrix
algebra:

```cpp
    Eigen::Matrix<std::complex<double>,2,1> y0;
    Eigen::Matrix<std::complex<double>,2,2> A;

    y0 << 1.0,
          1.0;

    A << 1.0, 0.0,
         0.0, -1.0;

    auto func = [&](double x, auto y, auto dy)
        {
            dy = A * y;
        };

    double last_x = 0;
    auto solout = [&] (double xa, double xb, const auto& sol) {
        if (xb == 0 || xb == 1 || xb > last_x + 0.2) {
            std::cout << xa << "..." << xb
                      << ": y(" << xb << ") = " << std::endl
                      << sol(xb)
                      << std::endl << std::endl;
            last_x = xb;
        }
        return false;
    };

    dopri5::solve(0, 1, y0, func, solout);
```

More can be found in [example.cpp](@ref example.cpp).  To build it,
and the reference documentation, run `cmake . && make`.
