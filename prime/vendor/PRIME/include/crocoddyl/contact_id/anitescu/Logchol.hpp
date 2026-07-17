///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
// University of Wisconsin-Madison
//
// This file implements log-Cholesky inertial-parameter utilities used by the
// contact-ID extension built on top of Crocoddyl. It is not part of upstream
// Crocoddyl.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <pinocchio/fwd.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/inertia.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <Eigen/Dense>

#include <vector>
#include <string>
#include <limits>
#include <iostream>
#include <stdexcept>

// ------------------------------------------------------------
// Compute U such that J ≈ U * U^T with U upper-triangular.
// (Uses a permutation trick since Eigen's LLT exposes L.)
// ------------------------------------------------------------
inline Eigen::Matrix4d computeUpperUU(const Eigen::Matrix4d &J)
{
    // permutation that reverses order: (0,1,2,3) -> (3,2,1,0)
    Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
    for (int i = 0; i < 4; ++i)
        P(i, 3 - i) = 1.0;

    const Eigen::Matrix4d J_hat = P * J * P.transpose();

    Eigen::LLT<Eigen::Matrix4d> llt(J_hat);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("[logchol_helper] Cholesky failed, J not SPD");

    const Eigen::Matrix4d L_hat = llt.matrixL();         // lower-triangular
    const Eigen::Matrix4d U = P.transpose() * L_hat * P; // upper-triangular in original ordering

    return U;
}

// ------------------------------------------------------------
// Compute 10D log-Cholesky parameter vector from a link inertia
// via PseudoInertia J = U U^T, with U upper-triangular.
// Ordering matches your comment:
//   [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]
// ------------------------------------------------------------
inline Eigen::Matrix<double, 10, 1>
computeLogCholeskyFromLink(const pinocchio::Model &model,
                           int j_link,
                           bool verbose = false)
{
    const pinocchio::Inertia &Ij = model.inertias.at(j_link);

    pinocchio::PseudoInertia Ji = Ij.toPseudoInertia();
    const Eigen::Matrix4d Ji_mat = Ji.toMatrix();

    if (verbose)
    {
        std::cout << "[logchol_helper] PseudoInertia from model:\n"
                  << Ji_mat << "\n";
    }

    Eigen::Matrix4d R = computeUpperUU(Ji_mat); // upper-triangular factor

    // Extract global scale alpha and normalize so R_scaled(3,3) = 1
    const double alpha = std::log(R(3, 3));
    const Eigen::Matrix4d R_scaled = R / R(3, 3);

    Eigen::Matrix<double, 10, 1> pi_log_chol;
    pi_log_chol(0) = alpha;                    // alpha
    pi_log_chol(1) = std::log(R_scaled(0, 0)); // d1
    pi_log_chol(2) = std::log(R_scaled(1, 1)); // d2
    pi_log_chol(3) = std::log(R_scaled(2, 2)); // d3

    pi_log_chol(4) = R_scaled(0, 1); // s12
    pi_log_chol(5) = R_scaled(1, 2); // s23
    pi_log_chol(6) = R_scaled(0, 2); // s13

    pi_log_chol(7) = R_scaled(0, 3); // t1
    pi_log_chol(8) = R_scaled(1, 3); // t2
    pi_log_chol(9) = R_scaled(2, 3); // t3

    if (verbose)
    {
        std::cout << "[logchol_helper] pi_log_chol:\n"
                  << pi_log_chol.transpose() << "\n";
    }

    return pi_log_chol;
}
