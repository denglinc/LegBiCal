///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//
// Contact-ID modifications:
//   Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
//   University of Wisconsin-Madison
//
// This file defines the robot-agnostic differentiable Anitescu contact-ID
// problem builder built on top of Crocoddyl.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_ID_PROBLEM_CONTACT_ANITESCU_ID_HPP_
#define CROCODDYL_CONTACT_ID_PROBLEM_CONTACT_ANITESCU_ID_HPP_

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/frame.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/spatial/se3.hpp>

#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/contact_id/actions/arrival-fwddyn-id.hpp"
#include "crocoddyl/contact_id/actions/contact-fwddyn-anitescu-id.hpp"
#include "crocoddyl/contact_id/actuations/floating-base-estimation.hpp"
#include "crocoddyl/contact_id/actuations/floating-base-id-arrivial.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/contact_id/config/contact-id-config.hpp"

namespace crocoddyl {

Eigen::VectorXd computeFrozenSFromLogCholeskyJacobian(
    const pinocchio::Model& model, pinocchio::JointIndex j_link, double alpha);

class ContactAnitescuIDProblem {
 public:
  ContactAnitescuIDProblem(const pinocchio::Model& rmodel,
                           const std::vector<pinocchio::JointIndex>& joints,
                           const std::vector<std::string>& contact_frame_names,
                           const ContactIDWeights& weights,
                           const std::string& force_log_path = "");
  virtual ~ContactAnitescuIDProblem();

  boost::shared_ptr<crocoddyl::ShootingProblem> createEstimationProblem(
      const Eigen::VectorXd& x0, const double timeStep,
      const std::size_t n_knots, std::vector<Eigen::VectorXd>& state_task,
      std::vector<Eigen::VectorXd>& ctrl_task);

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract> >
  createStepModels(const double timeStep, const std::size_t n_knots,
                   std::vector<Eigen::VectorXd>& state_task,
                   std::vector<Eigen::VectorXd>& ctrl_task);

  boost::shared_ptr<ActionModelAbstract> createDiscreteModel(
      const double timeStep, const Eigen::VectorXd& state_task,
      const Eigen::VectorXd& ctrl_task = Eigen::VectorXd());

  boost::shared_ptr<ActionModelAbstract> createArrivalModel();

  const Eigen::VectorXd& get_defaultState() const;
  const ContactIDWeights& get_weights() const;
  const std::string& get_force_log_path() const;

 public:
  pinocchio::Model rmodel_;
  pinocchio::Data rdata_;
  std::vector<pinocchio::JointIndex> joints_;
  std::vector<std::string> contact_frame_names_;
  boost::shared_ptr<StateMultibodyParams> state_;
  boost::shared_ptr<ActuationModelFloatingBaseEstimation> actuation_;

  std::vector<boost::shared_ptr<CostModelSum> > cost_models_;
  std::vector<boost::shared_ptr<ActionModelAbstract> > euler_models_;

  ContactIDWeights weights_;
  std::string force_log_path_;
  Eigen::VectorXd defaultstate_;
};

class QuadrupedAnitescuIDProblem : public ContactAnitescuIDProblem {
 public:
  QuadrupedAnitescuIDProblem(const pinocchio::Model& rmodel,
                             std::vector<pinocchio::JointIndex> joints,
                             std::vector<std::string> foot_names, char* path);
  QuadrupedAnitescuIDProblem(const pinocchio::Model& rmodel,
                             std::vector<pinocchio::JointIndex> joints,
                             std::vector<std::string> foot_names,
                             const ContactIDWeights& weights);
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACT_ID_PROBLEM_CONTACT_ANITESCU_ID_HPP_
