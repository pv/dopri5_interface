//
// example.cpp: examples using dopri5_interface.h
//

#include <iostream>
#include <string>

#include <Eigen/Core>

#define DOPRI5_INTERFACE_USE_EIGEN
#include "dopri5_interface.h"


void print_banner(std::string);


//
// Simple scalar function example
//

void my_func(double x, const double& y, double& dy)
{
    dy = -y;
}

void example_scalar()
{
    print_banner("example_scalar");

    std::vector<double> xs({0, 0.25, 0.5, 0.75, 1});
    std::vector<double> ys(xs.size());
    std::vector<double>::iterator yend;

    ys[0] = 1;
    yend = dopri5::solve_at(xs.begin(), xs.end(), ys.begin(), my_func);

    for (int i = 0; i < yend - ys.begin(); ++i) {
        std::cout << "x[" << i << "] = " << xs[i]
                  << "; y[" << i << "] = " << ys[i] << std::endl;
    }
}


//
// Simple array example, with a lambda
//

void example_array()
{
    print_banner("example_array");

    std::vector<double> xs({0, 0.25, 0.5, 0.75, 1});
    std::vector<double[2]> ys(xs.size());
    std::vector<double[2]>::iterator yend;

    auto func = [](double x, const auto& y, auto& dy) {
        dy[0] = y[0];
        dy[1] = -y[1];
    };

    ys[0][0] = 1;
    ys[0][1] = 2;
    yend = dopri5::solve_at(xs.begin(), xs.end(), ys.begin(), func);

    for (int i = 0; i < yend - ys.begin(); ++i) {
        std::cout << "x[" << i << "] = " << xs[i]
                  << "; y[" << i << "] = {"
                  << ys[i][0] << ", " << ys[i][1]
                  << "}" << std::endl;
    }
}


//
// Example using Eigen matrices
//

void example_eigen()
{
    print_banner("example_eigen");

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
}


//
// Functor example (compiler can inline the functor to the callback)
//

//!\private
struct my_functor_func {
    void operator()(double x, const double& y, double& dy)
        {
            dy = -y;
        }
};

//!\private
class my_functor_solout {
private:
    double last_x;

public:
    my_functor_solout() : last_x(0) {}

    bool operator()(double xa, double xb, const auto& sol)
        {
            if (xb == 0 || xb == 1 || xb > last_x + 0.2) {
                std::cout << "y(" << xb << ") = " << sol(xb)
                          << " (exact: " << exp(-xb) << ")" << std::endl;
                last_x = xb;
            }
            return false;
        }
};

void example_functors()
{
    double y0 = 1;
    my_functor_func mf;
    my_functor_solout ms;

    print_banner("example_functors");
    dopri5::solve(0, 1, y0, mf, ms);
}


//
// Entry point
//

void print_banner(std::string message)
{
    std::string banner;
    for (int i = 0; i < 79; ++i) banner += "-";

    std::cout << std::endl << banner << std::endl;
    std::cout << message << std::endl;
    std::cout << banner << std::endl;
}


int main()
{
    example_scalar();
    example_array();
    example_eigen();
    example_functors();
    return 0;
}
