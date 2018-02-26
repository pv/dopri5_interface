// -*-c++-*-
//
// dopri5_interface.h
//
// C++ interface to the DOPRI5 solver by Hairer & Wanner.
//

#ifndef DOPRI5_INTERFACE_H_
#define DOPRI5_INTERFACE_H_

#include <complex>
#include <vector>

#ifdef DOPRI5_INTERFACE_USE_EIGEN
#include <Eigen/Core>
#endif

#ifndef DOPRI5_INTERFACE_ROUTINE
#define DOPRI5_INTERFACE_ROUTINE dopri5_
#endif

#ifndef DOPRI5_INTERFACE_CONTD5
#define DOPRI5_INTERFACE_CONTD5 contd5_
#endif

#ifndef DOPRI5_INTERFACE_RESTRICT
#define DOPRI5_INTERFACE_RESTRICT __restrict__
#endif


//! dopri5 interface main namespace
namespace dopri5 {

    //! Exit status from the solver
    enum success_status {
        success = 1,
        interrupted = 2,
        fail_inconsistent_input = -1,
        fail_too_many_steps = -2,
        fail_too_small_step = -3,
        fail_too_stiff = -4
    };

    //! Solver parameters.
    //!
    //! Setting a value to 0 means using the default value.
    struct solver_parameters {
        //! Relative error tolerance
        double rtol = 1e-14;
        //! Absolute error tolerance
        double atol = 1e-14;
        //! Maximal step size (default: xend-x)
        double max_step = 0;
        //! Initial step size (default: automatic)
        double initial_step = 0;
        //! Maximal number of allowed steps (default: 100000)
        int max_num_steps = 0;
        //! Safety factor in step size prediction (default: 0.9)
        double safety = 0;
        //! The "beta" for stabilized step size control (default: 0.04)
        double beta = 0;
        //! \brief Test for stiffness is activated after step number
        //! `j*stiffness_check`.  If negative, not active. (default: 1000)
        int stiffness_check = 0;
        //! Whether to print debug output.
        bool verbose = false;
    };

    // Forward declaration of dense solution (see below)
    template <typename Vector>
    class dense_solution;

    //! \brief Implementation details \internal
    namespace detail {
        //! \brief Linearization and double-casting of data.
        //!
        //! The `ravel` class is responsible for accessing data in
        //! `typename Vector` as an array of double precision numbers.
        //!
        //! See comments on `scalar_ravel` below for details.
        template <typename Vector>
        class ravel;

        //! Const version of ravel for immutable data
        template <typename Vector>
        class const_ravel;

        //! Default callable type declarations
        template <typename Vector>
        struct default_types
        {
            typedef void func_type(double,
                                   typename const_ravel<Vector>::arg_type,
                                   typename ravel<Vector>::arg_type);

            typedef bool solout_type(double, double, const dense_solution<Vector> &);
        };
    }

    //! Representation for the solution of the differential equations within
    //! some interval.
    template <typename Vector>
    class dense_solution {
    private:
        const int m_nr;
        const double *m_y, *m_con, *m_xold, *m_h;
        const int *m_icomp, *m_nd;
        const Vector *m_y0;
        
    protected:
        //! Obtain one floating point component of the solution.
        double get_one(double x, int i) const;

    public:
        dense_solution(int nr, const double *y, const double *con,
                       const int *icomp, const int *nd,
                       const double *xold, const double *h,
                       const Vector *y0)
            : m_nr(nr), m_y(y), m_con(con), m_xold(xold), m_h(h),
              m_icomp(icomp), m_nd(nd), m_y0(y0)
            {}

        //! Store value of the solution at `x` into `y`
        void get(double x, Vector& y) const;

        //! Return the value of the solution at `x`
        typename detail::ravel<Vector>::return_type operator()(double x) const;
    };

    //
    // DOPRI5 external routine declarations
    //

    //! \internal
    namespace detail {
        extern "C" {
            //! DOPRI5 callback type
            typedef void fcn_type(const int *n,
                                  const double *x,
                                  const double *y,
                                  double *f,
                                  double *rpar,
                                  int *ipar);

            //! DOPRI5 solout callback type
            typedef void solout_type(const int *nr,
                                     const double *xold,
                                     const double *x,
                                     const double *y,
                                     const int *n,
                                     const double *con,
                                     const int *icomp,
                                     const int *nd,
                                     double *rpar,
                                     int *ipar,
                                     int *irtrn
#ifndef DOPRI5_INTERFACE_USE_ORIGINAL
                                     , const double *h
#endif
                );

            //! DOPRI5 continuous output routine
            double DOPRI5_INTERFACE_CONTD5(const int *ii,
                                           const double *x,
                                           const double *con,
                                           const int *icomp,
                                           const int *nd
#ifndef DOPRI5_INTERFACE_USE_ORIGINAL
                                           , const double *xold,
                                           const double *h
#endif
                );

            //! DOPRI5 main routine
            void DOPRI5_INTERFACE_ROUTINE(int *n,
                                          fcn_type *fcn,
                                          double *x,
                                          double *y,
                                          double *xend,
                                          double *rtol,
                                          double *atol,
                                          int *itol,
                                          solout_type *solout,
                                          int *iout,
                                          double *work,
                                          int *lwork,
                                          int *iwork,
                                          int *liwork,
                                          double *rpar,
                                          int *ipar,
                                          int *idid);
        }

        //! Function callback template for DOPRI5
        template <typename Vector, typename Fcn>
        void fcn(const int *DOPRI5_INTERFACE_RESTRICT n,
                 const double *DOPRI5_INTERFACE_RESTRICT x,
                 const double *DOPRI5_INTERFACE_RESTRICT y,
                 double *DOPRI5_INTERFACE_RESTRICT f,
                 double *DOPRI5_INTERFACE_RESTRICT rpar,
                 int *DOPRI5_INTERFACE_RESTRICT ipar)
        {
            void **param = reinterpret_cast<void **>(rpar);
            Fcn *callback = reinterpret_cast<Fcn *>(ipar);
            const Vector *DOPRI5_INTERFACE_RESTRICT y0 =
                reinterpret_cast<const Vector *>(param[0]);
            const_ravel<Vector> y_ravel(*n, y, y0);
            ravel<Vector> f_ravel(*n, f, y0);
            (*callback)(*x, y_ravel.obj(), f_ravel.obj());
        }

        //! Solout callback template for DOPRI5
        template <typename Vector, typename Solout>
        void solout(const int *DOPRI5_INTERFACE_RESTRICT nr,
                    const double *DOPRI5_INTERFACE_RESTRICT xold,
                    const double *DOPRI5_INTERFACE_RESTRICT x,
                    const double *DOPRI5_INTERFACE_RESTRICT y,
                    const int *DOPRI5_INTERFACE_RESTRICT n,
                    const double *DOPRI5_INTERFACE_RESTRICT con,
                    const int *DOPRI5_INTERFACE_RESTRICT icomp,
                    const int *DOPRI5_INTERFACE_RESTRICT nd,
                    double *DOPRI5_INTERFACE_RESTRICT rpar,
                    int *DOPRI5_INTERFACE_RESTRICT ipar,
                    int *DOPRI5_INTERFACE_RESTRICT irtrn
#ifndef DOPRI5_INTERFACE_USE_ORIGINAL
                    , const double *DOPRI5_INTERFACE_RESTRICT h
#endif
            )
        {
            void **param = reinterpret_cast<void **>(rpar);
            Solout *callback = reinterpret_cast<Solout *>(param[1]);
            const Vector *DOPRI5_INTERFACE_RESTRICT y0 =
                reinterpret_cast<const Vector *>(param[0]);
#ifndef DOPRI5_INTERFACE_USE_ORIGINAL
            dense_solution<Vector> sol(*nr, y, con, icomp, nd, xold, h, y0);
#else
            dense_solution<Vector> sol(*nr, y, con, icomp, nd, 0, 0, y0);
#endif
            const_ravel<Vector> y_ravel(*n, y, y0);
            bool do_exit = (*callback)(*x, *xold, y_ravel.obj(), sol);
            if (do_exit) {
                *irtrn = -1;
            }
        }
    }

    //! Solve a differential equation, producing dense output.
    //!
    //! \param x0  Start position for the solution.
    //! \param xend  Position to solve up to. If xend < x0, integration proceeds
    //!     to negative direction
    //! \param y0  Initial value of the solution vector.
    //!     Note that this value is modified by the routine.
    //! \param fcn  Callback function `f(double x, ConstVector& y, Vector& dy)`
    //!     providing the derivative
    //! \param solout  Callback function
    //!     `solout(double x, double xprev, ConstVector& y, const dense_output<Vector>& sol)`
    //!     to receive the dense output in the interval [xprev, x]
    //!     (or [x, xprev] if negative integration direction).
    //! \param params  Solver parameters.
    template <typename Vector,
              typename Fcn = typename detail::default_types<Vector>::func_type,
              typename Solout = typename detail::default_types<Vector>::solout_type>
    inline success_status solve(double x0, double xend, Vector& y0,
                                Fcn& fcn, Solout& solout,
                                solver_parameters params = {})
    {
        detail::ravel<Vector> y0_ravel(y0);

        int n = y0_ravel.size();

        int iout = 2;
        int liwork = n + 21;
        std::vector<int> iwork(liwork);
        int lwork = 8*n + 5*n + 21;
        std::vector<double> work(lwork);
        int itol = 0;

        double rtol = params.rtol;
        double atol = params.atol;

        std::fill(iwork.begin(), iwork.end(), 0);
        std::fill(work.begin(), work.end(), 0.0);
        work[1] = params.safety;
        work[4] = params.beta;
        work[5] = params.max_step;
        work[6] = params.initial_step;
        iwork[0] = params.max_num_steps;
        iwork[2] = params.verbose ? 0 : -1;
        iwork[3] = params.stiffness_check;
        iwork[4] = n;

        void *param[3] = {reinterpret_cast<void *>(&y0),
                          reinterpret_cast<void *>(&solout)};
        double *rpar = reinterpret_cast<double *>(param);
        int *ipar = reinterpret_cast<int *>(&fcn);
        int idid = 0;

        detail::DOPRI5_INTERFACE_ROUTINE(&n, &detail::fcn<Vector,Fcn>,
                                         &x0, y0_ravel.ptr(), &xend,
                                         &rtol, &atol, &itol,
                                         &detail::solout<Vector,Solout>,
                                         &iout,
                                         work.data(), &lwork,
                                         iwork.data(), &liwork,
                                         rpar, ipar, &idid);

        return static_cast<success_status>(idid);
    }

    //! \internal
    namespace detail {
        //! Solution getter (can be specialized)
        template <typename YIterator, typename Vector>
        inline void storage_get_solution(double x,
                                         const dense_solution<Vector> &sol,
                                         YIterator out)
        {
            sol.get(x, *out);
        }

        //! \brief Solout functor that stores values of the solution evaluated
        //! at specific points into an iterable.
        template <typename XIterator,
                  typename YIterator,
                  typename Vector = typename YIterator::value_type>
        class storage_solout
        {
        private:
            XIterator m_xpos, m_xend;
            YIterator m_ypos;

        public:
            storage_solout(XIterator xbegin, XIterator xend,
                           YIterator ybegin)
                : m_xpos(xbegin), m_xend(xend), m_ypos(ybegin)
                {}

            bool operator()(double x, double xold,
                            auto& value,
                            const dense_solution<Vector>& sol)
                {
                    while (m_xpos < m_xend) {
                        if ((xold <= x && !(*m_xpos <= x)) ||
                            (xold >= x && !(*m_xpos >= x))) {
                            break;
                        }
                        storage_get_solution(x, sol, m_ypos);
                        ++m_xpos;
                        ++m_ypos;
                    }
                    return !(m_xpos < m_xend);
                }

            const YIterator end() const { return m_ypos; }
        };
    }

    //! Solve a differential equation, storing the solution in an iterable.
    //!
    //! \param xbegin  Start for iterator containing x-coordinates.
    //! \param xend  End for iterator of x-coordinates.
    //! \param y0  Initial value of the solution vector.
    //!     Note that this value is modified by the routine.
    //! \param fcn  Callback function `f(double x, ConstVector& y, Vector& dy)`
    //!     providing the derivative.
    //! \param ybegin  Start for iterator to store solutions.
    //! \param params  Solver parameters.
    //! \return Iterator pointing after the last stored y-value.
    template <typename XIterator, typename YIterator, typename Vector,
              typename Fcn = typename detail::default_types<Vector>::func_type>
    inline YIterator solve_at(XIterator xbegin, XIterator xend,
                              Vector &y0, Fcn& fcn, YIterator ybegin,
                              solver_parameters params = {})
    {
        if (xbegin >= xend) {
            return ybegin;
        }
        detail::storage_solout<XIterator, YIterator, Vector> solout(xbegin, xend, ybegin);
        solve(*xbegin, *(xend - 1), y0, fcn, solout, params);
        return solout.end();
    }

    //
    // Function implementations
    //

    template <typename Vector>
    inline double dense_solution<Vector>::get_one(double x, int i) const
    {
        if (m_nr == 1) {
            // The first call to solout is with zero-width interval, and the
            // continuous output is not available.
            return m_y[i];
        }
        else {
            ++i;
#ifndef DOPRI5_INTERFACE_USE_ORIGINAL
            return detail::DOPRI5_INTERFACE_CONTD5(&i, &x, m_con, m_icomp, m_nd,
                                                   m_xold, m_h);
#else
            return detail::DOPRI5_INTERFACE_CONTD5(&i, &x, m_con, m_icomp, m_nd);
#endif
        }
    }

    template <typename Vector>
    inline void dense_solution<Vector>::get(double x, Vector &out) const
    {
        detail::ravel<Vector> out_ravel(out);
        double *p = out_ravel.ptr();
        for (size_t i = 0; i < out_ravel.size(); ++i) {
            p[i] = get_one(x, i);
        }
    }

    template <typename Vector>
    inline typename detail::ravel<Vector>::return_type
    dense_solution<Vector>::operator()(double x) const
    {
        typename detail::ravel<Vector>::return_type y(
            detail::ravel<Vector>::empty_like(m_y0));
        get(x, y);
        return y;
    }

    //
    // Data type support
    //

    //! \internal
    namespace detail {
        //
        // Scalars
        //

        //! Generic implementation of `ravel` for scalar data
        template <typename Scalar, typename Double>
        class scalar_ravel
        {
        protected:
            Scalar *m_obj;

        public:
            //! Argument type for the callback function
            typedef Scalar& arg_type;

            //! Return type for sol(x)
            typedef Scalar return_type;

            //! Initialize ravel from an object
            scalar_ravel(Scalar& obj) : m_obj(&obj) {}

            //! Initialize ravel from an array of double-precision numbers
            scalar_ravel(int n, Double *values, const Scalar *y0)
                : m_obj(reinterpret_cast<Scalar *>(values)) {}

            //! Return the number of double-precision numbers in the object
            size_t size() { return sizeof(Scalar) / sizeof(Double); }

            //! \brief Return a pointer to an array of the double-precision
            //! numbers in the object.
            Double* ptr() { return reinterpret_cast<Double *>(m_obj); }

            //! Return a reference to the object.
            arg_type obj() { return *m_obj; }

            //! Create a new object of the same size as `y0`
            static return_type empty_like(const Scalar *y0) { return 0; }
        };

        //! ravel for double-precision scalars
        template <>
        class ravel<double>
            : public scalar_ravel<double, double>
        {
            using scalar_ravel<double, double>::scalar_ravel;
        };

        //! const_ravel for double-precision scalars
        template <>
        class const_ravel<double>
            : public scalar_ravel<const double, const double>
        {
            using scalar_ravel<const double, const double>::scalar_ravel;
        };

        //! ravel for std::compled<double> scalars
        template <>
        class ravel<std::complex<double>>
            : public scalar_ravel<std::complex<double>, double>
        {
            using scalar_ravel<std::complex<double>, double>::scalar_ravel;
        };

        //! const_ravel for std::compled<double> scalars
        template <>
        class const_ravel<std::complex<double>>
            : public scalar_ravel<const std::complex<double>, const double>
        {
            using scalar_ravel<const std::complex<double>, const double>::scalar_ravel;
        };

        //
        // Fixed-size plain arrays
        //

        //! Generic implementation of `ravel` for plain fixed-size arrays
        template <typename Scalar, typename Double, size_t SZ>
        class array_ravel
        {
        protected:
            Scalar *m_obj;

        public:
            typedef Scalar*& arg_type;
            typedef Scalar* return_type;

            array_ravel(Scalar* obj) : m_obj(obj) {}

            array_ravel(int n, Double* values, const Scalar (*y0)[SZ])
                : m_obj(reinterpret_cast<Scalar *>(values)) {}

            size_t size() { return SZ * sizeof(Scalar) / sizeof(double); }

            Double* ptr() { return reinterpret_cast<Double *>(m_obj); }

            arg_type obj() { return m_obj; }

            // empty_like cannot be defined --- the return_type mismatch ensures
            // a compiler error.
            static return_type empty_like(const Scalar (*y0)[SZ]) {
                return 0;
            }
        };

        //! ravel for double-precision fixed-size arrays
        template <size_t SZ>
        class ravel<double[SZ]> : public array_ravel<double, double, SZ>
        {
            using array_ravel<double, double, SZ>::array_ravel;
        };

        //! const_ravel for double-precision fixed-size arrays
        template <size_t SZ>
        class const_ravel<double[SZ]>
            : public array_ravel<const double, const double, SZ>
        {
            using array_ravel<const double, const double, SZ>::array_ravel;
        };

        //! ravel for std::compled<double> fixed-size arrays
        template <size_t SZ>
        class ravel<std::complex<double>[SZ]>
            : public array_ravel<std::complex<double>, double, SZ>
        {
            using array_ravel<std::complex<double>, double, SZ>::array_ravel;
        };

        //! const_ravel for std::compled<double> fixed-size arrays
        template <size_t SZ>
        class const_ravel<std::complex<double>[SZ]>
            : public array_ravel<const std::complex<double>, const double, SZ>
        {
            using array_ravel<const std::complex<double>, const double, SZ>::array_ravel;
        };

#ifdef DOPRI5_INTERFACE_USE_EIGEN
        //
        // Eigen matrixes
        //

        //! Generic ravel for Eigen matrices.
        //!
        //! The matrix is represented as an array of double-precision numbers
        //! (and vice versa) using Eigen::Map.
        template <typename Double, typename Scalar, typename Matrix>
        class eigen_map_ravel
        {
        protected:
            typedef Eigen::Map<Matrix> map_type;
            map_type m_map;

        public:
            typedef map_type& arg_type;
            typedef Matrix return_type;

            eigen_map_ravel(Matrix& obj)
                : m_map(obj.data(), obj.rows(), obj.cols()) {}
            eigen_map_ravel(int n, Double *values, const Matrix *y0)
                : m_map(reinterpret_cast<Scalar *>(values),
                       y0->rows(),
                       y0->cols()) {}
            size_t size() { return m_map.size() * sizeof(Scalar) / sizeof(Double); }
            Double* ptr() { return reinterpret_cast<Double *>(m_map.data()); }
            arg_type obj() { return m_map; }

            static return_type empty_like(const Matrix *y0) {
                if (Matrix::RowsAtCompileTime != -1 &&
                    Matrix::ColsAtCompileTime != -1) {
                    Matrix y;
                    return y;
                }
                else {
                    Matrix y(y0->rows(), y0->cols());
                    return y;
                }
            }
        };

        template <int ROWS, int COLS, int OPTIONS, int MAXROWS, int MAXCOLS>
        class ravel<Eigen::Matrix<double,ROWS,COLS,OPTIONS,MAXROWS,MAXCOLS> >
            : public eigen_map_ravel<double, double, Eigen::Matrix<double,ROWS,COLS> >
        {
            using eigen_map_ravel<double, double, Eigen::Matrix<double,ROWS,COLS> >::eigen_map_ravel;
        };

        template <int ROWS, int COLS, int OPTIONS, int MAXROWS, int MAXCOLS>
        class const_ravel<Eigen::Matrix<double,ROWS,COLS,OPTIONS,MAXROWS,MAXCOLS> >
            : public eigen_map_ravel<const double, const double, const Eigen::Matrix<double,ROWS,COLS> >
        {
            using eigen_map_ravel<const double, const double, const Eigen::Matrix<double,ROWS,COLS> >::eigen_map_ravel;
        };

        template <int ROWS, int COLS, int OPTIONS, int MAXROWS, int MAXCOLS>
        class ravel<Eigen::Matrix<std::complex<double>,ROWS,COLS,OPTIONS,MAXROWS,MAXCOLS> >
            : public eigen_map_ravel<double, std::complex<double>, Eigen::Matrix<std::complex<double>,ROWS,COLS> >
        {
            using eigen_map_ravel<double, std::complex<double>, Eigen::Matrix<std::complex<double>,ROWS,COLS> >::eigen_map_ravel;
        };

        template <int ROWS, int COLS, int OPTIONS, int MAXROWS, int MAXCOLS>
        class const_ravel<Eigen::Matrix<std::complex<double>,ROWS,COLS,OPTIONS,MAXROWS,MAXCOLS> >
            : public eigen_map_ravel<const double, const std::complex<double>, const Eigen::Matrix<std::complex<double>,ROWS,COLS> >
        {
            using eigen_map_ravel<const double, const std::complex<double>, const Eigen::Matrix<std::complex<double>,ROWS,COLS> >::eigen_map_ravel;
        };
#endif // DOPRI5_INTERFACE_USE_EIGEN
    }
}

#endif // DOPRI5_INTERFACE_H_
