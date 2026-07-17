///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
// University of Wisconsin-Madison
//
// This file implements barrier SOCP utilities for differentiable Anitescu
// contact simulation in the contact-ID extension built on top of Crocoddyl.
// It is not part of upstream Crocoddyl.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#include <cmath>
#include "crocoddyl/contact_id/anitescu/NewtonSolver.hpp"

struct BarrierSOCP
{
    // Quadratic objective 0.5 * dv^T H dv + g^T dv.
    Eigen::MatrixXd H;
    Eigen::VectorXd g;

    double kappa = 50.0;
    double kappa_alpha = 5000.0;
    double mu = 1.0;

    // Small cone-coordinate regularization. This helps choose a stable contact
    // force distribution when several contacts are redundant or coplanar.
    double gamma_cone = 1e-6;

    // Per-contact second-order cone terms:
    //   alpha_i(dv) = a_i^T dv + c_i
    //   beta_i(dv)  = B_i dv
    //   s_i(dv)     = alpha_i^2 - ||beta_i||^2 > 0
    std::vector<Eigen::RowVectorXd> a_list;
    std::vector<Eigen::MatrixXd> B_list;
    std::vector<double> c_list;

    // Require alpha_i > 0 in feasibility checks, matching the physical cone
    // branch used for contact forces.
    bool enforce_alpha_positive = true;

    std::size_t m() const { return a_list.size(); }

    void clearConstraints()
    {
        a_list.clear();
        B_list.clear();
        c_list.clear();
    }

    void addConstraint(const Eigen::RowVectorXd &a, const Eigen::MatrixXd &B, double c)
    {
        if (H.rows() == 0)
            throw std::runtime_error("Set H/g before adding constraints.");
        if (a.cols() != H.cols())
            throw std::runtime_error("a has wrong nv.");
        if (B.cols() != H.cols())
            throw std::runtime_error("B has wrong nv.");
        a_list.push_back(a);
        B_list.push_back(B);
        c_list.push_back(c);
    }

    // Normal cone coordinate for contact i.
    inline double alpha(const Eigen::VectorXd &dv, std::size_t i) const
    {
        return a_list[i].dot(dv) + c_list[i];
    }

    // Tangential cone coordinates for contact i.
    inline Eigen::VectorXd beta(const Eigen::VectorXd &dv, std::size_t i) const
    {
        return B_list[i] * dv;
    }

    // Strict cone margin s_i(dv) = alpha_i^2 - ||beta_i||^2.
    inline double s(const Eigen::VectorXd &dv, std::size_t i) const
    {
        const double al = alpha(dv, i);
        const Eigen::VectorXd be = beta(dv, i);
        return al * al - be.squaredNorm();
    }

    bool feasible(const Eigen::VectorXd &dv) const
    {
        if (!dv.allFinite())
            return false;
        for (std::size_t i = 0; i < m(); ++i)
        {
            const double al = alpha(dv, i);
            const double si = s(dv, i);
            if (!(si > 0.0) || !std::isfinite(si))
                return false;
            if (enforce_alpha_positive && !(al > 0.0))
                return false;
        }
        return true;
    }

    // Build a strictly feasible Newton initial guess by projecting dv0 onto
    // conservative cone-coordinate equalities. The hard pass enforces both
    // alpha and beta targets; the fallback preserves alpha and penalizes beta
    // softly, which is useful for redundant multi-point foot contacts.
    Eigen::VectorXd feasibleInitEquality(
        const Eigen::VectorXd &dv0,
        double alpha_min = 1e-3,
        double reg_W = 1e-9,
        double reg_S = 1e-9,
        bool verbose = false,
        bool use_soft_beta_fallback = true,
        double rho_beta = 1e-2,
        double eta_beta = 1.0,
        int max_tries = 4,
        double s_margin = 1e-12
    ) const
    {
        const int nv = (int)H.cols();
        if ((int)dv0.size() != nv)
            throw std::runtime_error("dv0 size mismatch.");
        if (m() == 0)
            return dv0;

        const int t = (int)B_list[0].rows();

        auto print_feas_stats = [&](const Eigen::VectorXd &dv, const char *tag)
        {
            double min_alpha = 1e100, min_s = 1e100, max_beta = 0.0;
            std::size_t worst_i = 0;

            for (std::size_t i = 0; i < m(); ++i)
            {
                const double al = alpha(dv, i);
                const Eigen::VectorXd be = beta(dv, i);
                const double si = al * al - be.squaredNorm();
                const double bn = be.norm();

                if (al < min_alpha)
                    min_alpha = al;
                if (si < min_s)
                {
                    min_s = si;
                    worst_i = i;
                }
                if (bn > max_beta)
                    max_beta = bn;
            }

            std::cout << "[feasibleInitEquality][" << tag << "] "
                      << "feasible=" << (feasible(dv) ? "true" : "false")
                      << "  min_alpha=" << min_alpha
                      << "  min_s=" << min_s
                      << "  max||beta||=" << max_beta
                      << "  worst_i=" << worst_i
                      << std::endl;

            // Report the most restrictive cone constraint.
            {
                const double al = alpha(dv, worst_i);
                const Eigen::VectorXd be = beta(dv, worst_i);
                const double si = al * al - be.squaredNorm();
                std::cout << "    worst: alpha=" << al
                          << "  ||beta||=" << be.norm()
                          << "  s=" << si
                          << "  c=" << c_list[worst_i]
                          << std::endl;
            }
        };

        // First try a hard equality projection for both cone coordinates.
        // Increase margins and regularization if the projection is singular or
        // too close to the cone boundary.
        double alpha_min_try = alpha_min;
        double regW_try = reg_W;
        double regS_try = reg_S;

        Eigen::VectorXd dv_last = dv0;

        for (int attempt = 0; attempt < max_tries; ++attempt)
        {
            const int r = (int)m() * (1 + t);

            Eigen::MatrixXd Aeq(r, nv);
            Eigen::VectorXd beq(r);
            Aeq.setZero();
            beq.setZero();

            for (std::size_t i = 0; i < m(); ++i)
            {
                const int base = (int)i * (1 + t);

                // Alpha target leaves extra room for the current tangential
                // residual, improving the chance of strict feasibility.
                Aeq.row(base) = a_list[i];
                const double beta0 = (B_list[i] * dv0).norm();
                const double alpha_target = std::max(alpha_min_try, c_list[i]) + eta_beta * beta0;

                beq(base) = alpha_target - c_list[i];

                Aeq.block(base + 1, 0, t, nv) = B_list[i];
            }

            if (verbose)
            {
                Eigen::FullPivLU<Eigen::MatrixXd> lu(Aeq);
                std::cout << "[feasibleInitEquality] attempt=" << attempt
                          << "  hard(beta)=true"
                          << "  alpha_min=" << alpha_min_try
                          << "  regW=" << regW_try
                          << "  regS=" << regS_try
                          << "  rank(Aeq)=" << lu.rank()
                          << "  r=" << r << " nv=" << nv
                          << std::endl;
            }

            Eigen::MatrixXd W = H;
            W.diagonal().array() += regW_try;

            Eigen::LDLT<Eigen::MatrixXd> ldltW(W);
            if (ldltW.info() != Eigen::Success)
            {
                if (verbose)
                    std::cout << "  W factorization failed.\n";
                alpha_min_try *= 10.0;
                regW_try *= 10.0;
                regS_try *= 10.0;
                continue;
            }

            Eigen::MatrixXd AT = Aeq.transpose(); // nv x r
            Eigen::MatrixXd X = ldltW.solve(AT);

            Eigen::MatrixXd S = Aeq * X;
            S.diagonal().array() += regS_try;

            Eigen::LDLT<Eigen::MatrixXd> ldltS(S);
            if (ldltS.info() != Eigen::Success)
            {
                if (verbose)
                    std::cout << "  Schur factorization failed.\n";
                alpha_min_try *= 10.0;
                regW_try *= 10.0;
                regS_try *= 10.0;
                continue;
            }

            Eigen::VectorXd rhs = beq - Aeq * dv0;
            Eigen::VectorXd y = ldltS.solve(rhs);
            Eigen::VectorXd dv = dv0 + X * y;
            dv_last = dv;

            if (verbose)
            {
                const Eigen::VectorXd res = Aeq * dv - beq;
                std::cout << "  eq_res_inf=" << res.lpNorm<Eigen::Infinity>()
                          << "  eq_res_2=" << res.norm()
                          << std::endl;
                print_feas_stats(dv, "hard-beta");
            }

            // Use a positive numerical margin, not only exact feasibility.
            bool ok = dv.allFinite();
            for (std::size_t i = 0; i < m(); ++i)
            {
                const double al = alpha(dv, i);
                const double si = s(dv, i);
                if (!(si > s_margin) || !std::isfinite(si))
                    ok = false;
                if (enforce_alpha_positive && !(al > std::sqrt(s_margin)))
                    ok = false;
            }

            if (ok)
                return dv;

            alpha_min_try *= 10.0;
            regW_try *= 10.0;
            regS_try *= 10.0;
        }

        // Fallback: keep alpha constraints hard and regularize tangential
        // coordinates through the quadratic metric.
        if (use_soft_beta_fallback)
        {
            const int rA = (int)m();

            Eigen::MatrixXd Aalpha(rA, nv);
            Eigen::VectorXd balpha(rA);
            Aalpha.setZero();
            balpha.setZero();

            for (std::size_t i = 0; i < m(); ++i)
            {
                Aalpha.row((int)i) = a_list[i];

                const double beta0 = (B_list[i] * dv0).norm();
                const double alpha_target = std::max(alpha_min, c_list[i]) + eta_beta * beta0;
                balpha((int)i) = alpha_target - c_list[i];
            }

            Eigen::MatrixXd W = H;
            W.diagonal().array() += reg_W;

            if (rho_beta > 0.0)
            {
                for (std::size_t i = 0; i < m(); ++i)
                    W.noalias() += rho_beta * (B_list[i].transpose() * B_list[i]);
            }

            if (verbose)
            {
                std::cout << "[feasibleInitEquality] fallback soft-beta: "
                          << "rho_beta=" << rho_beta
                          << "  alpha_min=" << alpha_min
                          << "  rank(Aalpha)=" << Eigen::FullPivLU<Eigen::MatrixXd>(Aalpha).rank()
                          << "  rA=" << rA << " nv=" << nv
                          << std::endl;
            }

            Eigen::LDLT<Eigen::MatrixXd> ldltW(W);
            if (ldltW.info() != Eigen::Success)
            {
                if (verbose)
                    std::cout << "  W_eff factorization failed in fallback.\n";
                return dv_last;
            }

            Eigen::MatrixXd AT = Aalpha.transpose(); // nv x rA
            Eigen::MatrixXd X = ldltW.solve(AT);     // nv x rA
            Eigen::MatrixXd S = Aalpha * X;          // rA x rA
            S.diagonal().array() += reg_S;

            Eigen::LDLT<Eigen::MatrixXd> ldltS(S);
            if (ldltS.info() != Eigen::Success)
            {
                if (verbose)
                    std::cout << "  Schur factorization failed in fallback.\n";
                return dv_last;
            }

            Eigen::VectorXd rhs = balpha - Aalpha * dv0;
            Eigen::VectorXd y = ldltS.solve(rhs);
            Eigen::VectorXd dv = dv0 + X * y;

            if (verbose)
            {
                const Eigen::VectorXd res = Aalpha * dv - balpha;
                std::cout << "  alpha_eq_res_inf=" << res.lpNorm<Eigen::Infinity>()
                          << "  alpha_eq_res_2=" << res.norm()
                          << std::endl;
                print_feas_stats(dv, "soft-beta");
            }

            if (!feasible(dv))
            {
                if (verbose)
                    std::cout << "  soft-beta fallback remains infeasible; returning best effort.\n";
                return dv;
            }

            return dv;
        }

        return dv_last;
    }

    double f(const Eigen::VectorXd &dv) const
    {
        double val = 0.5 * dv.dot(H * dv) + g.dot(dv);

        // Minimum-norm regularization in cone coordinates.
        if (gamma_cone > 0.0)
        {
            for (std::size_t i = 0; i < m(); ++i)
            {
                const double al = alpha(dv, i);
                const Eigen::VectorXd be = beta(dv, i);
                val += 0.5 * gamma_cone * (al * al + be.squaredNorm());
            }
        }

        for (std::size_t i = 0; i < m(); ++i)
        {
            const double si = s(dv, i);
            val -= (1.0 / kappa) * std::log(si);
        }
        return val;
    }

    void grad(const Eigen::VectorXd &dv, Eigen::VectorXd &out) const
    {
        out = H * dv + g;

        // Gradient of the cone-coordinate quadratic regularization.
        if (gamma_cone > 0.0)
        {
            for (std::size_t i = 0; i < m(); ++i)
            {
                const auto &a = a_list[i];
                const auto &B = B_list[i];
                const double al = a.dot(dv) + c_list[i];
                const Eigen::VectorXd be = B * dv;
                out += gamma_cone * (al * a.transpose() + B.transpose() * be);
            }
        }

        for (std::size_t i = 0; i < m(); ++i)
        {
            const auto &a = a_list[i];
            const auto &B = B_list[i];

            const double al = a.dot(dv) + c_list[i];
            const Eigen::VectorXd be = B * dv;
            const double si = al * al - be.squaredNorm();

            Eigen::VectorXd grad_s = 2.0 * al * a.transpose() - 2.0 * (B.transpose() * be);
            out += -(1.0 / kappa) * (grad_s / si);
        }
    }

    void hess(const Eigen::VectorXd &dv, Eigen::MatrixXd &out) const
    {
        out = H;

        // Hessian of the cone-coordinate quadratic regularization.
        if (gamma_cone > 0.0)
        {
            for (std::size_t i = 0; i < m(); ++i)
            {
                const auto &a = a_list[i];
                const auto &B = B_list[i];
                out += gamma_cone * (a.transpose() * a + B.transpose() * B);
            }
        }

        for (std::size_t i = 0; i < m(); ++i)
        {
            const auto &a = a_list[i];
            const auto &B = B_list[i];

            const double al = a.dot(dv) + c_list[i];
            const Eigen::VectorXd be = B * dv;

            const double si = al * al - be.squaredNorm();

            Eigen::VectorXd grad_s = 2.0 * al * a.transpose() - 2.0 * (B.transpose() * be);
            Eigen::MatrixXd Hs = 2.0 * (a.transpose() * a) - 2.0 * (B.transpose() * B);

            out += (1.0 / kappa) * ((grad_s * grad_s.transpose()) / (si * si) - (Hs / si));
        }
    }

    // Scalar barrier weight per cone. This is useful for diagnostics, but it is
    // not a full SOCP dual vector.
    Eigen::VectorXd lambda_weight(const Eigen::VectorXd &dv) const
    {
        Eigen::VectorXd w(m());
        for (std::size_t i = 0; i < m(); ++i)
            w((int)i) = (1.0 / kappa) * (1.0 / s(dv, i));
        return w;
    }

    Eigen::VectorXd lambda_forces(const Eigen::VectorXd &dv) const
    {
        const std::size_t m_c = m();
        if (m_c == 0)
        {
            return Eigen::VectorXd();
        }

        const int t = static_cast<int>(B_list[0].rows());
        const int block = 1 + t;
        Eigen::VectorXd out(m_c * block);
        out.setZero();

        for (std::size_t i = 0; i < m_c; ++i)
        {
            const double al = alpha(dv, i);
            const Eigen::VectorXd be = beta(dv, i);
            const double si = s(dv, i);

            const double lambda_i = 1.0 / (kappa * si);

            const double f_n = 2.0 * lambda_i * al / mu;
            const Eigen::VectorXd f_t = -2.0 * lambda_i * be;

            const int base = static_cast<int>(i) * block;
            out.segment(base, t) = f_t;
            out(base + t) = f_n;
        }
        return out;
        }
};
