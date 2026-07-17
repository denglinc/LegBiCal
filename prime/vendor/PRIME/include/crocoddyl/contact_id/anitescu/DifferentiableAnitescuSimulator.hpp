///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
// University of Wisconsin-Madison
//
// This file implements differentiable Anitescu contact simulation utilities for
// contact estimation and inertial-parameter identification. It is part of the
// contact-ID extension built on top of Crocoddyl, but is not part of upstream
// Crocoddyl.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/regressor.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <limits>

// Barrier contact solve and Newton line-search utilities.
#include "crocoddyl/contact_id/anitescu/BarrierSOCP.hpp"
#include "crocoddyl/contact_id/anitescu/NewtonSolver.hpp"
#include "crocoddyl/contact_id/anitescu/Logchol.hpp"

static inline Eigen::Matrix3d skew3(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d S;
    S << 0.0, -v.z(), v.y(),
        v.z(), 0.0, -v.x(),
        -v.y(), v.x(), 0.0;
    return S;
}

static inline Eigen::Matrix<double, 3, 2> tangentBasisFromNormal(const Eigen::Vector3d &n_in)
{
    Eigen::Vector3d n = n_in.normalized();
    Eigen::Vector3d ref = (std::abs(n.z()) < 0.9) ? Eigen::Vector3d::UnitZ()
                                                  : Eigen::Vector3d::UnitX();
    Eigen::Vector3d t1 = ref - n * (n.dot(ref));
    double nt1 = t1.norm();
    if (nt1 < 1e-12)
    {
        ref = Eigen::Vector3d::UnitY();
        t1 = ref - n * (n.dot(ref));
        nt1 = t1.norm();
    }
    t1 /= nt1;
    Eigen::Vector3d t2 = n.cross(t1);
    Eigen::Matrix<double, 3, 2> T;
    T.col(0) = t1;
    T.col(1) = t2;
    return T;
}

struct StepResult
{
    Eigen::VectorXd v_next;    // nv
    Eigen::MatrixXd dv_dq;     // nv x nv
    Eigen::MatrixXd dv_dv;     // nv x nv
    Eigen::MatrixXd dv_dtau;   // nv x nv
    Eigen::MatrixXd dv_dtheta; // nv x ntheta

    // Diagnostic outputs from the contact solve and sensitivity pass.
    Eigen::MatrixXd H;            // nv x nv (prob.H)
    Eigen::VectorXd g;            // nv
    Eigen::VectorXd sum_g;        // nv
    Eigen::MatrixXd sum_dg_dq;    // nv x nv
    Eigen::VectorXd sum_cone;     // nv
    Eigen::MatrixXd sum_dcone_dq; // nv x nv
    Eigen::VectorXd force;        // 3*m (contact forces)
    bool newton_converged = false;
    int newton_iterations = 0;
    double newton_gradient_norm = std::numeric_limits<double>::infinity();
    double min_cone_margin = -std::numeric_limits<double>::infinity();
};

class DifferentiableAnitescuSimulator
{
public:
    struct Options
    {
        double mu = 0.3;
        double plane_height = 0.0;
        double kappa = 50.0;
        double damping = 0.05;
        double armature = 0.01;
        // When provided, these vectors replace the legacy floating-base
        // convention (scalar regularization with the first six entries zero).
        // This is required for planar bases and for measured motor armatures.
        Eigen::VectorXd damping_vector;
        Eigen::VectorXd armature_vector;
        Eigen::Vector3d n_vec = Eigen::Vector3d(0, 0, 1);
        bool log_chol = true;

        bool enforce_alpha_positive = true;
        bool verbose = false;

        NewtonSolver::Options newton;
    };

    DifferentiableAnitescuSimulator(const pinocchio::Model &model,
                                    const std::vector<std::string> &contact_frame_names,
                                    const Options &opt)
        : model_(model), opt_(opt), data_(model)
    {
        nv_ = model_.nv;
        nq_ = model_.nq;

        // Contact tangent basis associated with the configured ground normal.
        T_ = tangentBasisFromNormal(opt_.n_vec);

        // Resolve contact frames once; per-step kinematics refreshes placements.
        setContacts(contact_frame_names);

        // Global dynamics buffers.
        M_.setZero(nv_, nv_);
        h_.setZero(nv_);
        a_free_.setZero(nv_);
        v_free_next_.setZero(nv_);

        // RNEA derivative workspaces.
        drnea_dq_.setZero(nv_, nv_);
        drnea_dv_.setZero(nv_, nv_);
        drnea_da_.setZero(nv_, nv_);

        // Sensitivity outputs.
        dv_dq_.setZero(nv_, nv_);

        // Kinematic workspaces reused across contacts.
        J6_.setZero(6, nv_);
        Jp_.setZero(3, nv_);
        lwaJf_.setZero(6, nv_);

        // Armature and damping are disabled on the floating-base coordinates.
        armature_ = opt_.armature;
        damping_ = opt_.damping;
        M_armature_.setZero(nv_, nv_);
        D_damping_.setZero(nv_, nv_);
        if (opt_.armature_vector.size() == nv_)
            M_armature_.diagonal() = opt_.armature_vector;
        else
        {
            M_armature_.diagonal().setConstant(armature_);
            M_armature_.diagonal().head(std::min(6, nv_)).setZero();
        }
        if (opt_.damping_vector.size() == nv_)
            D_damping_.diagonal() = opt_.damping_vector;
        else
        {
            D_damping_.diagonal().setConstant(damping_);
            D_damping_.diagonal().head(std::min(6, nv_)).setZero();
        }

        // Per-contact storage depends on the resolved contact set.
        resizeContactWorkspaces();

        solver_ = NewtonSolver(opt_.newton);
    }

    // Reconfigure the ordered contact frame set.
    void setContacts(const std::vector<std::string> &contact_frame_names)
    {
        contact_names_ = contact_frame_names;
        const std::size_t m = contact_names_.size();
        frame_ids_.resize(m);
        joint_ids_.resize(m);

        for (std::size_t i = 0; i < m; ++i)
        {
            const auto fid = model_.getFrameId(contact_names_[i]);
            if (fid == (pinocchio::FrameIndex)(-1))
                throw std::runtime_error("Contact frame not found: " + contact_names_[i]);
            frame_ids_[i] = fid;
            joint_ids_[i] = model_.frames[fid].parentJoint;
        }
        resizeContactWorkspaces();
    }

    // Run one differentiable Anitescu contact step.
    StepResult step(const Eigen::VectorXd &q,
                    const Eigen::VectorXd &v,
                    const Eigen::VectorXd &tau,
                    double dt)
    {
        if (q.size() != nq_ || v.size() != nv_ || tau.size() != nv_)
            throw std::runtime_error("Dimension mismatch in step().");

        dt_ = dt;
        q_ = q;
        v_ = v;
        tau_ = tau;

        // Free dynamics without contact impulses.
        computeFreeDynamics();

        // Contact geometry, Jacobians, and Hessian transfer terms.
        updateKinematics(q_);
        buildContactsFromFrames();

        // Barrier problem in next generalized velocity.
        buildBarrierProblem();

        // Newton solve and analytic sensitivities.
        solveBarrierNewton();
        computeSensitivities();

        Eigen::VectorXd force = prob_.lambda_forces(v_next_) / dt_;

        StepResult out;
        out.v_next = v_next_;
        out.dv_dq = dv_dq_;
        out.dv_dv = dv_dv_;
        out.dv_dtau = dv_dtau_;
        out.dv_dtheta = dv_dtheta_;
        out.H = prob_.H;
        out.g = prob_.g;
        out.sum_g = sum_g_;
        out.sum_dg_dq = sum_dg_dq_;
        out.sum_cone = sum_cone_;
        out.sum_dcone_dq = sum_dcone_dq_;
        out.force = force;
        out.newton_converged = newton_converged_;
        out.newton_iterations = solver_.lastIterations();
        out.newton_gradient_norm = solver_.lastGradientNorm();
        out.min_cone_margin = min_cone_margin_;
        return out;
    }

    // Accessors for callers that need Pinocchio data or contact derivatives.
    pinocchio::Data &data() { return data_; }
    const pinocchio::Data &data() const { return data_; }

    const std::vector<Eigen::MatrixXd> &dg_dq_list() const { return dg_dq_list_; }

private:
    // Free forward dynamics with armature and damping regularization.
    void computeFreeDynamics()
    {
        pinocchio::computeAllTerms(model_, data_, q_, v_);
        M_ = data_.M;
        M_.template triangularView<Eigen::StrictlyLower>() = M_.transpose();
        M_ = M_ + M_armature_;
        h_ = pinocchio::rnea(model_, data_, q_, v_, Eigen::VectorXd::Zero(nv_));
        Eigen::LDLT<Eigen::MatrixXd> mass_ldlt(M_);
        if (mass_ldlt.info() != Eigen::Success)
            throw std::runtime_error("LDLT failed for the regularized mass matrix.");
        a_free_ = mass_ldlt.solve(tau_ - h_);
        v_free_next_.noalias() = v_ + dt_ * a_free_;
    }

    // Refresh frame placements, joint Jacobians, and kinematic Hessians.
    void updateKinematics(const Eigen::VectorXd &q)
    {
        pinocchio::forwardKinematics(model_, data_, q);
        pinocchio::updateFramePlacements(model_, data_);
        pinocchio::computeJointJacobians(model_, data_);
        pinocchio::computeJointKinematicHessians(model_, data_, q);
    }

    // Build per-contact signed distance, normal/tangential Jacobians, and
    // Hessian transfer terms. The current implementation uses each contact
    // frame origin as the contact point.
    void buildContactsFromFrames()
    {
        const std::size_t m = frame_ids_.size();
        for (std::size_t ci = 0; ci < m; ++ci)
        {
            const pinocchio::FrameIndex fid = frame_ids_[ci];
            const pinocchio::JointIndex jid = joint_ids_[ci];

            const pinocchio::SE3 &oMf = data_.oMf[fid];
            p_world_[ci] = oMf.translation();

            Phi_[ci] = opt_.n_vec.dot(p_world_[ci]) - opt_.plane_height;

            // Convert the joint spatial Jacobian to a point linear Jacobian.
            J6_.setZero();
            pinocchio::getJointJacobian(model_, data_, jid, pinocchio::LOCAL_WORLD_ALIGNED, J6_);

            const Eigen::Vector3d oMj = data_.oMi[jid].translation();
            const Eigen::Matrix3d oRj = data_.oMi[jid].rotation();
            const Eigen::Vector3d r = p_world_[ci] - oMj;

            Jp_.noalias() = J6_.topRows(3) - skew3(r) * J6_.bottomRows(3);

            Jn_[ci].noalias() = opt_.n_vec.transpose() * Jp_;
            Jt_[ci].noalias() = T_.transpose() * Jp_;

            // Arguments for transferring joint Hessians to the contact point.
            lwaJf_.topRows(3) = Jp_;
            lwaJf_.bottomRows(3) = J6_.bottomRows(3);

            const Eigen::Vector3d p_j = oRj.transpose() * (p_world_[ci] - oMj);
            jMf_ = pinocchio::SE3(Eigen::Matrix3d::Identity(), p_j);
            oRf_ = oRj;

            transferKinematicsHessian(jid,
                                      lwaJf_,
                                      jMf_,
                                      oRf_,
                                      opt_.n_vec, T_,
                                      lwaHf_linear_[ci],
                                      lwaHf_linear_normal_[ci],
                                      lwaHf_linear_tangent_[ci],
                                      lwaHf_linear_stack_[ci],
                                      lwaHf_linear_normal_stack_[ci],
                                      lwaHf_linear_tangent_stack_[ci]);
        }
    }

    // Assemble the per-step cone barrier problem.
    void buildBarrierProblem()
    {
        prob_.H = M_ + dt_ * D_damping_;
        prob_.g = -M_ * v_free_next_;
        prob_.kappa = opt_.kappa;
        prob_.mu = opt_.mu;
        prob_.enforce_alpha_positive = opt_.enforce_alpha_positive;
        prob_.clearConstraints();

        const double mu = opt_.mu;
        const std::size_t m = frame_ids_.size();

        for (std::size_t ci = 0; ci < m; ++ci)
        {
            const Eigen::RowVectorXd a = (1.0 / mu) * Jn_[ci];
            const Eigen::MatrixXd B = Jt_[ci];

            double c = (Phi_[ci] / dt_) * (1.0 / mu);
            c += 1e-6;
            prob_.addConstraint(a, B, c);
        }
    }

    // Solve the cone barrier problem with a feasible Newton initialization.
    void solveBarrierNewton()
    {
        v_next_.setZero(nv_);
        Eigen::VectorXd v_init =
            (has_cached_solution_ && prob_.feasible(cached_v_next_))
                ? cached_v_next_
                : v_;

        if (!prob_.feasible(v_init))
        {
            v_init = prob_.feasibleInitEquality(v_init, /*alpha_min=*/1e-2);
        }

        v_next_ = v_init;

        newton_converged_ = solver_.solve(
            v_next_,
            [&](const Eigen::VectorXd &x)
            { return prob_.f(x); },
            [&](const Eigen::VectorXd &x, Eigen::VectorXd &gg)
            { prob_.grad(x, gg); },
            [&](const Eigen::VectorXd &x, Eigen::MatrixXd &HH)
            { prob_.hess(x, HH); },
            [&](const Eigen::VectorXd &x)
            { return prob_.feasible(x); });

        if (newton_converged_)
        {
            cached_v_next_ = v_next_;
            has_cached_solution_ = true;
        }

        min_cone_margin_ = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < prob_.m(); ++i)
            min_cone_margin_ = std::min(min_cone_margin_, prob_.s(v_next_, i));
    }

    // Differentiate the optimality equation. Contact barrier derivatives are
    // accumulated per contact, then solved through the barrier Hessian.
    void computeSensitivities()
    {
        compute_dg_dq_allContacts();

        Eigen::MatrixXd H_next;
        prob_.hess(v_next_, H_next);
        Eigen::LLT<Eigen::MatrixXd> llt(H_next);
        if (llt.info() != Eigen::Success)
            throw std::runtime_error("LLT failed: prob.H not SPD.");

        drnea_dq_.setZero(nv_, nv_);
        drnea_dv_.setZero(nv_, nv_);
        drnea_da_.setZero(nv_, nv_);

        pinocchio::computeRNEADerivatives(model_, data_,
                                          q_, v_, (v_next_ - v_) / dt_,
                                          drnea_dq_, drnea_dv_, drnea_da_);

        sum_dg_dq_.setZero(nv_, nv_);
        for (auto &G : dg_dq_list_)
            sum_dg_dq_.noalias() += G;

        sum_g_.setZero(nv_);
        for (auto &G : g_list_)
            sum_g_.noalias() += G;

        sum_dcone_dq_.setZero(nv_, nv_);
        for (auto &G : dcone_dq_list_)
            sum_dcone_dq_.noalias() += G;

        sum_cone_.setZero(nv_);
        for (auto &G : cone_list_)
            sum_cone_.noalias() += G;

        dv_dq_.noalias() = -llt.solve(dt_ * drnea_dq_ + sum_dg_dq_);
        dv_dv_.noalias() = -llt.solve(-M_ + dt_ * drnea_dv_);
        dv_dtau_.noalias() = -llt.solve(-dt_ * Eigen::MatrixXd::Identity(nv_, nv_));

        const std::size_t nj = joints_stack_.size();
        Eigen::MatrixXd Ysel(model_.nv, 10 * nj);
        Ysel.setZero();

        if (!opt_.log_chol)
        {
            // Linear dynamic-parameter sensitivity from the torque regressor.
            pinocchio::computeJointTorqueRegressor(model_, data_, q_, v_, (v_next_ - v_) / dt_);
            const Eigen::MatrixXd &Yfull = data_.jointTorqueRegressor; // nv x 10*(njoints-1)

            for (std::size_t j = 0; j < nj; ++j)
            {
                pinocchio::JointIndex jid = joints_stack_[j];
                if (jid == 0 || jid >= model_.njoints)
                    throw std::runtime_error("jid out of range (note: joint 0 is universe).");

                const int col0_full = 10 * int(jid - 1);
                const int col0_sel = 10 * int(j);

                Ysel.block(0, col0_sel, model_.nv, 10) =
                    Yfull.block(0, col0_full, model_.nv, 10);
            }
        }
        else
        {
            // Log-Cholesky parameter sensitivity via the dynamic-parameter
            // regressor and the analytic d(pi)/d(eta) map.
            pinocchio::computeJointTorqueRegressor(model_, data_, q_, v_, (v_next_ - v_) / dt_);
            const Eigen::MatrixXd &Yfull = data_.jointTorqueRegressor; // nv x 10*(njoints-1)

            for (std::size_t j = 0; j < nj; ++j)
            {

                pinocchio::JointIndex jid = joints_stack_[j];
                if (jid == 0 || jid >= model_.njoints)
                    throw std::runtime_error("jid out of range (note: joint 0 is universe).");

                Eigen::VectorXd p_jid = computeLogCholeskyFromLink(model_, jid);

                pinocchio::LogCholeskyParametersTpl<double> log_chol_jid(p_jid);
                Eigen::Matrix<double, 10, 10> dpi_dpi_log_chol_jid = log_chol_jid.calculateJacobian();

                const int col0_full = 10 * int(jid - 1);
                const int col0_sel = 10 * int(j);
                Ysel.block(0, col0_sel, model_.nv, 10) =
                    Yfull.block(0, col0_full, model_.nv, 10) *
                    dpi_dpi_log_chol_jid;
            }
        }
        Eigen::MatrixXd dv_dtheta_sel(model_.nv, 10 * nj);
        dv_dtheta_sel.noalias() = -llt.solve(dt_ * Ysel);
        dv_dtheta_ = dv_dtheta_sel;
    }

    void compute_dg_dq_allContacts()
    {
        const double mu = opt_.mu;
        const double inv_mu2 = 1.0 / (mu * mu);
        const std::size_t m = frame_ids_.size();

        for (std::size_t ci = 0; ci < m; ++ci)
        {
            const Eigen::RowVectorXd &Jn = Jn_[ci];
            const Eigen::MatrixXd &Jt = Jt_[ci];
            const double Phi = Phi_[ci];

            const Eigen::VectorXd &vplus = v_next_;

            const double s = (Phi / dt_ + (Jn * vplus)(0, 0));
            const double alpha = inv_mu2 * s;
            const Eigen::Vector2d beta = Jt * vplus;

            cone_.noalias() = alpha * Jn.transpose() - Jt.transpose() * beta;

            dalpha_dq_ = inv_mu2 * (Jn / dt_);
            for (int k = 0; k < nv_; ++k)
                dalpha_dq_(0, k) += inv_mu2 * (lwaHf_linear_normal_stack_[ci][k] * vplus)(0, 0);

            dcone_normal_dq_.noalias() = Jn.transpose() * dalpha_dq_;
            for (int k = 0; k < nv_; ++k)
                dcone_normal_dq_.col(k) += alpha * lwaHf_linear_normal_stack_[ci][k].transpose();

            dbeta_dq_.setZero(2, nv_);
            for (int k = 0; k < nv_; ++k)
                dbeta_dq_.col(k) = lwaHf_linear_tangent_stack_[ci][k] * vplus;

            dcone_tangent_dq_.setZero(nv_, nv_);
            for (int j = 0; j < nv_; ++j)
            {
                const Eigen::MatrixXd &dJt = lwaHf_linear_tangent_stack_[ci][j];
                const Eigen::VectorXd t1 = dJt.transpose() * beta;
                const Eigen::VectorXd t2 = Jt.transpose() * dbeta_dq_.col(j);
                dcone_tangent_dq_.col(j) = t1 + t2;
            }

            dcone_dq_ = dcone_normal_dq_ - dcone_tangent_dq_;

            const double cone2 = inv_mu2 * (s * s) - beta.squaredNorm();
            dcone2_dq_.noalias() = 2.0 / inv_mu2 * alpha * dalpha_dq_ - 2.0 * beta.transpose() * dbeta_dq_;

            const double cc = -(2.0 / opt_.kappa) * (1.0 / cone2);
            const double inv_cone2_sq = 1.0 / (cone2 * cone2);

            g_ = -(2.0 / (opt_.kappa * cone2)) * cone_;

            dg_dq_.setZero(nv_, nv_);
            for (int k = 0; k < nv_; ++k)
            {
                const double dc_dqk = (2.0 / opt_.kappa) * inv_cone2_sq * dcone2_dq_(0, k);
                dg_dq_.col(k) = dc_dqk * cone_ + cc * dcone_dq_.col(k);
            }
            dg_dq_list_[ci] = dg_dq_;
            g_list_[ci] = g_;
            cone_list_[ci] = cone_;
            dcone_dq_list_[ci] = dcone_dq_;
        }
    }

    // Transfer Pinocchio's joint kinematic Hessian to the contact point and
    // project it into world-aligned linear, normal, and tangential components.
    void transferKinematicsHessian(pinocchio::JointIndex joint_id,
                                   const Eigen::MatrixXd &lwaJf,
                                   const pinocchio::SE3 &jMf,
                                   const Eigen::Matrix3d &oRf,
                                   const Eigen::Vector3d &normal_contact,
                                   const Eigen::Matrix<double, 3, 2> &tangent_contact,
                                   Eigen::Tensor<double, 3> &lwaHf_linear,
                                   Eigen::Tensor<double, 3> &lwaHf_linear_normal,
                                   Eigen::Tensor<double, 3> &lwaHf_linear_tangent,
                                   std::vector<Eigen::MatrixXd> &lwaHf_linear_stack,
                                   std::vector<Eigen::MatrixXd> &lwaHf_linear_normal_stack,
                                   std::vector<Eigen::MatrixXd> &lwaHf_linear_tangent_stack)
    {
        const int nv = nv_;

        lwaHf_linear.setZero();
        lwaHf_linear_normal.setZero();
        lwaHf_linear_tangent.setZero();

        Eigen::Tensor<double, 3> jHj(6, nv, nv);
        jHj.setZero();
        pinocchio::getJointKinematicHessian(model_, data_, joint_id, pinocchio::LOCAL, jHj);

        const pinocchio::SE3::ActionMatrixType fXj = jMf.inverse().toActionMatrix();

        lwaHf_linear_stack.resize(nv);
        lwaHf_linear_normal_stack.resize(nv);
        lwaHf_linear_tangent_stack.resize(nv);

        for (int i = 0; i < nv; ++i)
        {
            Eigen::Tensor<double, 2> t_jHj_i = jHj.chip(i, 2).eval();
            Eigen::MatrixXd jHj_i = Eigen::Map<const Eigen::MatrixXd>(t_jHj_i.data(), 6, nv);

            Eigen::MatrixXd fHf_i = fXj * jHj_i;

            Eigen::MatrixXd lwaHf_i = oRf * fHf_i.topRows<3>();

            // Account for the point offset in the linear Hessian.
            Eigen::Matrix3d S;
            pinocchio::skew(lwaJf.bottomRows<3>().col(i), S);
            lwaHf_i.noalias() += S * lwaJf.topRows<3>();

            Eigen::MatrixXd lwaHf_normal_i = normal_contact.transpose() * lwaHf_i;
            Eigen::MatrixXd lwaHf_tangent_i = tangent_contact.transpose() * lwaHf_i;

            lwaHf_linear_stack[i] = lwaHf_i;
            lwaHf_linear_normal_stack[i] = lwaHf_normal_i;
            lwaHf_linear_tangent_stack[i] = lwaHf_tangent_i;

            // Keep both tensor and matrix-stack forms for callers that use
            // either representation.
            Eigen::TensorMap<Eigen::Tensor<double, 2>> t_lwaHf(lwaHf_i.data(), 3, nv);
            lwaHf_linear.chip(i, 2) = t_lwaHf;

            Eigen::TensorMap<Eigen::Tensor<double, 2>> t_n(lwaHf_normal_i.data(), 1, nv);
            lwaHf_linear_normal.chip(i, 2) = t_n;

            Eigen::TensorMap<Eigen::Tensor<double, 2>> t_t(lwaHf_tangent_i.data(), 2, nv);
            lwaHf_linear_tangent.chip(i, 2) = t_t;
        }
    }

    // Allocate per-contact buffers after contact-frame changes.
    void resizeContactWorkspaces()
    {
        const std::size_t m = contact_names_.size();

        p_world_.assign(m, Eigen::Vector3d::Zero());
        Phi_.assign(m, 0.0);

        Jn_.assign(m, Eigen::RowVectorXd::Zero(nv_));
        Jt_.assign(m, Eigen::MatrixXd::Zero(2, nv_));

        dg_dq_list_.assign(m, Eigen::MatrixXd::Zero(nv_, nv_));
        g_list_.assign(m, Eigen::VectorXd::Zero(nv_));
        cone_list_.assign(m, Eigen::VectorXd::Zero(nv_));
        dcone_dq_list_.assign(m, Eigen::MatrixXd::Zero(nv_, nv_));

        lwaHf_linear_.assign(m, Eigen::Tensor<double, 3>(3, nv_, nv_));
        lwaHf_linear_normal_.assign(m, Eigen::Tensor<double, 3>(1, nv_, nv_));
        lwaHf_linear_tangent_.assign(m, Eigen::Tensor<double, 3>(2, nv_, nv_));

        lwaHf_linear_stack_.assign(m, std::vector<Eigen::MatrixXd>(nv_, Eigen::MatrixXd::Zero(3, nv_)));
        lwaHf_linear_normal_stack_.assign(m, std::vector<Eigen::MatrixXd>(nv_, Eigen::MatrixXd::Zero(1, nv_)));
        lwaHf_linear_tangent_stack_.assign(m, std::vector<Eigen::MatrixXd>(nv_, Eigen::MatrixXd::Zero(2, nv_)));

        sum_dg_dq_.setZero(nv_, nv_);
    }

public:
    std::vector<pinocchio::JointIndex> joints_stack_;

private:
    // Model and options owned by the caller/configuration layer.
    const pinocchio::Model &model_;
    Options opt_;

    // Internal Pinocchio data reused across steps.
    pinocchio::Data data_;

    int nv_{0}, nq_{0};

    // Cached contact frame and parent-joint identifiers.
    std::vector<std::string> contact_names_;
    std::vector<pinocchio::FrameIndex> frame_ids_;
    std::vector<pinocchio::JointIndex> joint_ids_;

    // Contact tangent basis.
    Eigen::Matrix<double, 3, 2> T_;

    // Current-step input snapshot.
    double dt_{0.0};
    Eigen::VectorXd q_, v_, tau_;

    // Dynamics workspaces.
    Eigen::MatrixXd M_;
    Eigen::VectorXd h_, a_free_, v_free_next_;

    Eigen::MatrixXd M_armature_;
    Eigen::MatrixXd D_damping_;

    double damping_{0.05};
    double armature_{0.01};

    // Cone barrier problem and Newton solver.
    BarrierSOCP prob_;
    NewtonSolver solver_;
    bool newton_converged_{false};
    double min_cone_margin_{-std::numeric_limits<double>::infinity()};

    // Solved next generalized velocity.
    Eigen::VectorXd v_next_;
    Eigen::VectorXd cached_v_next_;
    bool has_cached_solution_{false};

    // RNEA derivatives at the current step.
    Eigen::MatrixXd drnea_dq_, drnea_dv_, drnea_da_;

    // Step sensitivities returned through StepResult.
    Eigen::MatrixXd dv_dq_;
    Eigen::MatrixXd dv_dv_;
    Eigen::MatrixXd dv_dtau_;
    Eigen::MatrixXd dv_dtheta_;

    // Per-contact geometry and Jacobians.
    std::vector<Eigen::Vector3d> p_world_;
    std::vector<double> Phi_;
    std::vector<Eigen::RowVectorXd> Jn_;
    std::vector<Eigen::MatrixXd> Jt_;

    // Per-contact Hessian transfer containers.
    std::vector<Eigen::Tensor<double, 3>> lwaHf_linear_;
    std::vector<Eigen::Tensor<double, 3>> lwaHf_linear_normal_;
    std::vector<Eigen::Tensor<double, 3>> lwaHf_linear_tangent_;

    std::vector<std::vector<Eigen::MatrixXd>> lwaHf_linear_stack_;
    std::vector<std::vector<Eigen::MatrixXd>> lwaHf_linear_normal_stack_;
    std::vector<std::vector<Eigen::MatrixXd>> lwaHf_linear_tangent_stack_;

    // Per-contact barrier-gradient derivatives and their sums.
    std::vector<Eigen::MatrixXd> dg_dq_list_;
    std::vector<Eigen::VectorXd> g_list_;
    Eigen::MatrixXd sum_dg_dq_;
    Eigen::VectorXd sum_g_;

    std::vector<Eigen::MatrixXd> dcone_dq_list_;
    std::vector<Eigen::VectorXd> cone_list_;
    Eigen::MatrixXd sum_dcone_dq_;
    Eigen::VectorXd sum_cone_;

    // Reused kinematic work matrices.
    Eigen::MatrixXd J6_;
    Eigen::MatrixXd Jp_;
    Eigen::MatrixXd lwaJf_;
    pinocchio::SE3 jMf_;
    Eigen::Matrix3d oRf_;

    // Barrier differentiation temporaries.
    Eigen::RowVectorXd dalpha_dq_;
    Eigen::MatrixXd dcone_dq_;
    Eigen::MatrixXd dcone_normal_dq_;
    Eigen::MatrixXd dcone_tangent_dq_;
    Eigen::MatrixXd dbeta_dq_;
    Eigen::RowVectorXd dcone2_dq_;
    Eigen::VectorXd g_;
    Eigen::MatrixXd dg_dq_;

    Eigen::VectorXd cone_;
};
