///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
// University of Wisconsin-Madison
//
// This file implements a lightweight Newton solver used by the differentiable
// Anitescu contact simulator in the contact-ID extension. It is not part of
// upstream Crocoddyl.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <Eigen/Dense>
#include <limits>

class NewtonSolver {
public:
  struct Options {
    int max_iters;
    double grad_tol;
    double step_tol;
    double c1;          // Armijo
    double alpha_init;
    double alpha_min;
    double backtrack;   // alpha *= backtrack
    double reg;         // Levenberg reg if Hessian not SPD
    bool verbose;

    Options()
      : max_iters(100),
        grad_tol(1e-8),
        step_tol(1e-10),
        c1(1e-4),
        alpha_init(1.0),
        alpha_min(1e-8),
        backtrack(0.5),
        reg(1e-9),
        verbose(false) {}
  };

  NewtonSolver() : opt_() {}
  explicit NewtonSolver(const Options& opt) : opt_(opt) {}

  int lastIterations() const { return last_iterations_; }
  double lastGradientNorm() const { return last_gradient_norm_; }
  double lastStepNorm() const { return last_step_norm_; }

  // -------- templated solve (no std::function) --------
  template<typename F, typename G, typename HFun, typename Feas>
  bool solve(Eigen::VectorXd& x,
             F&& f,
             G&& g,
             HFun&& H,
             Feas&& feasible)
  {
    const int n = x.size();
    Eigen::VectorXd grad(n);
    Eigen::MatrixXd hess(n, n);

    last_iterations_ = 0;
    last_gradient_norm_ = std::numeric_limits<double>::infinity();
    last_step_norm_ = std::numeric_limits<double>::infinity();

    double fx = f(x);

    for (int it = 0; it < opt_.max_iters; ++it) {
      g(x, grad);
      double gn = grad.norm();
      last_iterations_ = it + 1;
      last_gradient_norm_ = gn;

      if (opt_.verbose) {
        std::cout << "[Newton] it " << it
                  << "  f=" << fx
                  << "  ||g||=" << gn << "\n";
      }

      if (gn < opt_.grad_tol) {
        last_step_norm_ = 0.0;
        return true;
      }

      H(x, hess);

      Eigen::VectorXd step;
      if (!solveSPD(hess, grad, step)) {
        Eigen::MatrixXd hreg =
            hess + opt_.reg * Eigen::MatrixXd::Identity(n, n);
        if (!solveSPD(hreg, grad, step)) return false;
      }

      step = -step;
      last_step_norm_ = step.norm();
      if (last_step_norm_ < opt_.step_tol) return true;

      double alpha = lineSearch(x, step, fx, grad, f, feasible);
      if (alpha < opt_.alpha_min) return false;

      x.noalias() += alpha * step;
      fx = f(x);
    }
    return false;
  }

  // Overload without feasible() (always feasible)
  template<typename F, typename G, typename HFun>
  bool solve(Eigen::VectorXd& x,
             F&& f,
             G&& g,
             HFun&& H)
  {
    auto always_feasible = [](const Eigen::VectorXd&) { return true; };
    return solve(x, std::forward<F>(f),
                    std::forward<G>(g),
                    std::forward<HFun>(H),
                    always_feasible);
  }

private:
  Options opt_;
  int last_iterations_{0};
  double last_gradient_norm_{std::numeric_limits<double>::infinity()};
  double last_step_norm_{std::numeric_limits<double>::infinity()};

  bool solveSPD(const Eigen::MatrixXd& H,
                const Eigen::VectorXd& g,
                Eigen::VectorXd& sol) const
  {
    Eigen::LLT<Eigen::MatrixXd> llt(H);
    if (llt.info() != Eigen::Success) return false;
    sol = llt.solve(g);
    return (llt.info() == Eigen::Success);
  }

  template<typename F, typename Feas>
  double lineSearch(const Eigen::VectorXd& x,
                    const Eigen::VectorXd& p,
                    double fx,
                    const Eigen::VectorXd& grad,
                    F&& f,
                    Feas&& feasible)
  {
    double alpha = opt_.alpha_init;
    double slope = grad.dot(p); // should be negative

    Eigen::VectorXd xt(x.size());

    while (alpha > opt_.alpha_min) {
      xt.noalias() = x + alpha * p;

      if (!feasible(xt)) {
        alpha *= opt_.backtrack;
        continue;
      }

      double ft = f(xt);
      if (ft <= fx + opt_.c1 * alpha * slope) return alpha;

      alpha *= opt_.backtrack;
    }
    return alpha;
  }
};
