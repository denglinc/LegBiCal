///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//
// Contact-ID modifications:
//   Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
//   University of Wisconsin-Madison
//
// This file implements the robot-agnostic differentiable Anitescu contact-ID
// problem builder built on top of Crocoddyl.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/contact_id/problem/contact-anitescu-id.hpp"

#include <cmath>

#include <pinocchio/algorithm/joint-configuration.hpp>

#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/contact_id/anitescu/Logchol.hpp"

namespace crocoddyl {

Eigen::VectorXd computeFrozenSFromLogCholeskyJacobian(
    const pinocchio::Model& model, pinocchio::JointIndex j_link, double alpha) {
  Eigen::VectorXd eta = computeLogCholeskyFromLink(model, j_link);
  pinocchio::LogCholeskyParametersTpl<double> log_chol(eta);

  Eigen::VectorXd pi = log_chol.toDynamicParameters();
  const double m = pi[0];
  if (!(m > 0.) || !std::isfinite(m)) {
    throw std::runtime_error("Mass is not positive/finite at theta0.");
  }

  Eigen::Matrix<double, 10, 10> dpi_deta = log_chol.calculateJacobian();
  Eigen::Matrix<double, 1, 10> dm_deta = dpi_deta.row(0);
  Eigen::VectorXd g = (dm_deta.array() / m).abs().matrix().transpose();
  return (1. + alpha * g.array().square()).sqrt().matrix();
}

ContactAnitescuIDProblem::ContactAnitescuIDProblem(
    const pinocchio::Model& rmodel,
    const std::vector<pinocchio::JointIndex>& joints,
    const std::vector<std::string>& contact_frame_names,
    const ContactIDWeights& weights, const std::string& force_log_path)
    : rmodel_(rmodel),
      rdata_(rmodel_),
      joints_(joints),
      contact_frame_names_(contact_frame_names),
      state_(boost::make_shared<crocoddyl::StateMultibodyParams>(
          boost::make_shared<pinocchio::Model>(rmodel_), joints_)),
      actuation_(
          boost::make_shared<crocoddyl::ActuationModelFloatingBaseEstimation>(
              state_)),
      weights_(weights),
      force_log_path_(force_log_path),
      defaultstate_(rmodel_.nq + rmodel_.nv + 2 * 10 * joints_.size()) {
  if (joints_.empty()) {
    throw std::runtime_error("ContactAnitescuIDProblem requires at least one identified joint.");
  }
  defaultstate_.head(rmodel_.nq) =
      rmodel_.referenceConfigurations.count("standing")
          ? rmodel_.referenceConfigurations["standing"]
          : pinocchio::neutral(rmodel_);
  defaultstate_.tail(rmodel_.nv + 2 * 10 * joints_.size()).setZero();
}

ContactAnitescuIDProblem::~ContactAnitescuIDProblem() {}

boost::shared_ptr<crocoddyl::ShootingProblem>
ContactAnitescuIDProblem::createEstimationProblem(
    const Eigen::VectorXd& x0, const double timestep,
    const std::size_t n_knots, std::vector<Eigen::VectorXd>& state_task,
    std::vector<Eigen::VectorXd>& ctrl_task) {
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >
      estimation3d_model;
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >
      estimation_step;

  estimation3d_model.push_back(createArrivalModel());
  estimation_step = createStepModels(timestep, n_knots, state_task, ctrl_task);

  estimation3d_model.insert(estimation3d_model.end(), estimation_step.begin(),
                            estimation_step.end() - 1);
  return boost::make_shared<crocoddyl::ShootingProblem>(
      x0, estimation3d_model, estimation_step.back());
}

std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >
ContactAnitescuIDProblem::createStepModels(
    double timestep, const std::size_t n_knots,
    std::vector<Eigen::VectorXd>& state_task,
    std::vector<Eigen::VectorXd>& ctrl_task) {
  std::vector<boost::shared_ptr<ActionModelAbstract> > step_model;
  for (std::size_t i = 0; i < n_knots; ++i) {
    if (i >= state_task.size()) {
      throw std::runtime_error("Not enough state task rows for requested knots.");
    }
    if (i < ctrl_task.size()) {
      step_model.push_back(createDiscreteModel(timestep, state_task[i],
                                               ctrl_task[i]));
    } else {
      step_model.push_back(createDiscreteModel(timestep, state_task[i]));
    }
    euler_models_.push_back(step_model[i]);
  }
  return step_model;
}

boost::shared_ptr<crocoddyl::ActionModelAbstract>
ContactAnitescuIDProblem::createDiscreteModel(
    double timestep, const Eigen::VectorXd& state_task,
    const Eigen::VectorXd& ctrl_task) {
  boost::shared_ptr<crocoddyl::CostModelSum> cost_model =
      boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());

  if (state_task.array().allFinite()) {
    Eigen::VectorXd state_weights(state_->get_ndx());
    state_weights.segment(0, 2).fill(std::pow(weights_.meas_position, 2));
    state_weights.segment(2, 1).fill(std::pow(weights_.meas_position_z, 2));
    state_weights.segment(3, 3).fill(std::pow(weights_.meas_orientation, 2));
    state_weights.segment(6, rmodel_.nv - 6)
        .fill(std::pow(weights_.meas_joint, 2));
    state_weights.segment(rmodel_.nv, state_->get_np())
        .fill(std::pow(weights_.meas_params, 2));
    state_weights.segment(state_->get_nq(), 3)
        .fill(std::pow(weights_.meas_linearVel, 2));
    state_weights.segment(state_->get_nq() + 3, 3)
        .fill(std::pow(weights_.meas_angularVel, 2));
    state_weights.segment(state_->get_nq() + 6, rmodel_.nv - 6)
        .fill(std::pow(weights_.meas_jointVel, 2));
    state_weights.segment(state_->get_ndx() - state_->get_np(),
                          state_->get_np())
        .setZero();

    boost::shared_ptr<crocoddyl::ActivationModelAbstract> state_activation =
        boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(
            state_weights);
    boost::shared_ptr<crocoddyl::CostModelAbstract> state_reg =
        boost::make_shared<crocoddyl::CostModelResidual>(
            state_, state_activation,
            boost::make_shared<crocoddyl::ResidualModelState>(
                state_, state_task, actuation_->get_nu()));
    cost_model->addCost("stateReg", state_reg, 1e1);
  }

  if (ctrl_task.size() > 0 && ctrl_task.array().allFinite()) {
    Eigen::VectorXd ctrl_weights(actuation_->get_nu());
    ctrl_weights.segment(0, 3).fill(std::pow(weights_.dyn_position, 2));
    ctrl_weights.segment(3, 3).fill(std::pow(weights_.dyn_orientation, 2));
    ctrl_weights.segment(6, actuation_->get_nu() - 6)
        .fill(std::pow(weights_.dyn_joint, 2));

    boost::shared_ptr<crocoddyl::ActivationModelAbstract> ctrl_activation =
        boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(
            ctrl_weights);
    boost::shared_ptr<crocoddyl::CostModelAbstract> ctrl_reg =
        boost::make_shared<crocoddyl::CostModelResidual>(
            state_, ctrl_activation,
            boost::make_shared<crocoddyl::ResidualModelControl>(state_,
                                                                ctrl_task));
    cost_model->addCost("ctrlReg", ctrl_reg, 1e1);
  }

  cost_models_.push_back(cost_model);

  boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dmodel =
      boost::make_shared<
          crocoddyl::DifferentialActionModelContactFwdDynamicsAnitescuSystemid>(
          state_, actuation_, cost_model, contact_frame_names_, weights_.kappa,
          weights_.mu, timestep, force_log_path_);

  return boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel,
                                                                   timestep);
}

boost::shared_ptr<crocoddyl::ActionModelAbstract>
ContactAnitescuIDProblem::createArrivalModel() {
  boost::shared_ptr<ActuationModelFloatingBaseArrivalID> actuation_arrival =
      boost::make_shared<crocoddyl::ActuationModelFloatingBaseArrivalID>(state_);

  const int nu = static_cast<int>(actuation_arrival->get_nu());
  boost::shared_ptr<crocoddyl::CostModelSum> cost_model =
      boost::make_shared<crocoddyl::CostModelSum>(state_, nu);

  Eigen::VectorXd defect_task = Eigen::VectorXd::Zero(nu);
  Eigen::VectorXd defect_weights = Eigen::VectorXd::Zero(nu);

  const int stride = 10;
  if (nu % stride != 0) {
    throw std::runtime_error(
        "Arrival control dimension is not a multiple of 10.");
  }
  const int n_links_id = nu / stride;
  for (int j = 0; j < n_links_id; ++j) {
    const int base = j * stride;
    defect_weights(base + 0) = std::pow(weights_.arrival_alpha, 2);
    defect_weights(base + 1) = std::pow(weights_.arrival_diag, 2);
    defect_weights(base + 2) = std::pow(weights_.arrival_diag, 2);
    defect_weights(base + 3) = std::pow(weights_.arrival_diag, 2);
    defect_weights(base + 4) = std::pow(weights_.arrival_diag, 2);
    defect_weights(base + 5) = std::pow(weights_.arrival_diag, 2);
    defect_weights(base + 6) = std::pow(weights_.arrival_diag, 2);
    defect_weights(base + 7) = std::pow(weights_.arrival_com, 2);
    defect_weights(base + 8) = std::pow(weights_.arrival_com, 2);
    defect_weights(base + 9) = std::pow(weights_.arrival_com, 2);
  }

  Eigen::VectorXd s_all(nu);
  for (int j = 0; j < n_links_id; ++j) {
    Eigen::VectorXd s_link = computeFrozenSFromLogCholeskyJacobian(
        rmodel_, joints_[static_cast<std::size_t>(j)],
        weights_.arrival_scale_alpha);
    if (s_link.size() != stride) {
      throw std::runtime_error(
          "Log-Cholesky scaling must return size 10 per link.");
    }
    s_all.segment(j * stride, stride) = s_link;
  }
  Eigen::VectorXd weights = defect_weights.array() * s_all.array().square();

  boost::shared_ptr<crocoddyl::ActivationModelAbstract> defect_activation =
      boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights);
  boost::shared_ptr<crocoddyl::CostModelAbstract> defect_reg =
      boost::make_shared<crocoddyl::CostModelResidual>(
          state_, defect_activation,
          boost::make_shared<crocoddyl::ResidualModelControl>(state_,
                                                              defect_task));
  cost_model->addCost("ctrlReg", defect_reg, 1e1);

  return boost::make_shared<crocoddyl::ActionModelArrivalFwdDynamicsID>(
      state_, actuation_arrival, cost_model);
}

const Eigen::VectorXd& ContactAnitescuIDProblem::get_defaultState() const {
  return defaultstate_;
}

const ContactIDWeights& ContactAnitescuIDProblem::get_weights() const {
  return weights_;
}

const std::string& ContactAnitescuIDProblem::get_force_log_path() const {
  return force_log_path_;
}

QuadrupedAnitescuIDProblem::QuadrupedAnitescuIDProblem(
    const pinocchio::Model& rmodel, std::vector<pinocchio::JointIndex> joints,
    std::vector<std::string> foot_names, char* path)
    : ContactAnitescuIDProblem(
          rmodel, joints, foot_names,
          contact_id_xml::load_legacy_weights_file(path ? std::string(path)
                                                        : std::string())) {}

QuadrupedAnitescuIDProblem::QuadrupedAnitescuIDProblem(
    const pinocchio::Model& rmodel, std::vector<pinocchio::JointIndex> joints,
    std::vector<std::string> foot_names, const ContactIDWeights& weights)
    : ContactAnitescuIDProblem(rmodel, joints, foot_names, weights) {}

}  // namespace crocoddyl
