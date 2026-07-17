///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh, CTU, INRIA,
//                          University of Oxford, Heriot-Watt University
//
// Contact-ID modifications:
//   Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
//   University of Wisconsin-Madison
//
// This file is derived from and extends Crocoddyl contact forward-dynamics
// action interfaces for differentiable Anitescu contact estimation and
// inertial-parameter identification.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>

#include <pinocchio/algorithm/regressor.hpp>
#include <pinocchio/spatial/inertia.hpp>

#include "crocoddyl/contact_id/actions/contact-fwddyn-anitescu-id.hpp"

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/utils/csv_to_eigen.hpp"
#include "crocoddyl/multibody/utils/rapidxml.hpp"

using namespace rapidxml;
static inline double read_attr_double(xml_node<> *parent, const char *child_name,
                                      const char *attr_name, double defval = 0.0)
{
  if (!parent)
    return defval;
  if (auto *child = parent->first_node(child_name))
  {
    if (auto *attr = child->first_attribute(attr_name))
    {
      try
      {
        return std::stod(attr->value());
      }
      catch (...)
      { /* fall through */
      }
    }
  }
  return defval;
}

namespace crocoddyl
{
  template <typename Scalar>
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::
      DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl(
          boost::shared_ptr<StateMultibodyParams> state,
          boost::shared_ptr<ActuationModelAbstract> actuation,
          boost::shared_ptr<CostModelSum> costs,
          std::vector<std::string> contact_frames,
          const Scalar kappa,
          const Scalar mu,
          const Scalar dt,
          const std::string &force_log_path)
      : Base(state, actuation->get_nu(), costs->get_nr(), 0, 0),
        actuation_(actuation),
        costs_(costs),
        constraints_(nullptr),
        pinocchio_(*state->get_pinocchio().get()),
        with_armature_(true),
        armature_(VectorXs::Zero(state->get_nv_pin())),
        contact_frames_(contact_frames),
        kappa_(kappa),
        mu_(mu),
        dt_(dt),
        force_log_path_(force_log_path),
        params_state_(state)
  {
    init();
  }

  template <typename Scalar>
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::
      DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl(
          boost::shared_ptr<StateMultibodyParams> state,
          boost::shared_ptr<ActuationModelAbstract> actuation,
          boost::shared_ptr<CostModelSum> costs,
          boost::shared_ptr<ConstraintModelManager> constraints,
          std::vector<std::string> contact_frames,
          const Scalar kappa,
          const Scalar mu,
          const Scalar dt,
          const std::string &force_log_path)
      : Base(state, actuation->get_nu(), costs->get_nr(), constraints->get_ng(),
             constraints->get_nh(), constraints->get_ng_T(),
             constraints->get_nh_T()),
        actuation_(actuation),
        costs_(costs),
        constraints_(constraints),
        pinocchio_(*state->get_pinocchio().get()),
        with_armature_(true), // true means that we didn't set the armature
        armature_(VectorXs::Zero(state->get_nv_pin())),
        contact_frames_(contact_frames),
        kappa_(kappa),
        mu_(mu),
        dt_(dt),
        force_log_path_(force_log_path),
        params_state_(state)
  {
    init();
  }

  template <typename Scalar>
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::
      ~DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl() {}

  template <typename Scalar>
  void DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::init()
  {

    if (costs_->get_nu() != nu_)
    {
      throw_pretty(
          "Invalid argument: "
          << "Costs doesn't have the same control dimension (it should be " +
                 std::to_string(nu_) + ")");
    }

    VectorXs u_lb = Scalar(-1.) * pinocchio_.effortLimit.tail(nu_);
    VectorXs u_ub = Scalar(+1.) * pinocchio_.effortLimit.tail(nu_);
    Base::set_u_lb(u_lb);
    Base::set_u_ub(u_ub);
  }

  template <typename Scalar>
  void DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::calc(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != params_state_->get_nx())
    {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(params_state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_)
    {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }

    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
        x.head(params_state_->get_nq_pin());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
        x.segment(params_state_->get_nq(), params_state_->get_nv_pin());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> p =
        x.segment(params_state_->get_nq_pin(), params_state_->get_np());

    // Debug use only
    const VectorXs q_dbg = x.head(params_state_->get_nq_pin()).eval();
    const VectorXs v_dbg = x.segment(params_state_->get_nq(), params_state_->get_nv_pin()).eval();
    const VectorXs p_dbg = x.segment(params_state_->get_nq_pin(), params_state_->get_np());

    actuation_->calc(d->multibody.actuation, x, u);

    // Assigning the inertias from the parameter vector
    std::vector<pinocchio::JointIndex> joints = params_state_->get_param_joints();
    for (std::size_t jiter = 0; jiter < joints.size(); ++jiter)
    {

      pinocchio::JointIndex jid = joints[jiter];
      VectorXs p_jid = p.segment(10 * jiter, 10);
      pinocchio::LogCholeskyParametersTpl<Scalar> log_chol_jid(p_jid);
      VectorXs pi = log_chol_jid.toDynamicParameters(); // <-- if available
      pinocchio::Inertia I = pinocchio::Inertia::FromDynamicParameters(pi);

      pinocchio_.inertias[jid] = I;
    }

    //--------------------------------------------------------------------------------------------
    // Computing the forward dynamics using contact simulator
    //--------------------------------------------------------------------------------------------
    pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);

    DifferentiableAnitescuSimulator::Options opts;
    opts.mu = mu_;
    opts.plane_height = 0.0;
    opts.kappa = kappa_;
    opts.armature_vector = armature_.template cast<double>();
    opts.damping_vector = Eigen::VectorXd::Zero(pinocchio_.nv);

    anitescu_sim_ = std::make_unique<DifferentiableAnitescuSimulator>(pinocchio_, contact_frames_, opts);
    anitescu_sim_->joints_stack_ = joints;

    Eigen::VectorXd q_double = q.template cast<double>();
    Eigen::VectorXd v_double = v.template cast<double>();
    Eigen::VectorXd tau_double = d->multibody.actuation->tau.template cast<double>();

    anitescu_sim_out_ = std::make_unique<StepResult>();
    *anitescu_sim_out_ = anitescu_sim_->step(q_double, v_double, tau_double, dt_);
    Eigen::VectorXd force = anitescu_sim_out_->force;
    if (!force_log_path_.empty())
    {
      csvutil::logVectorToCSV(force, force_log_path_);
    }

    VectorXs a_constr = VectorXs::Zero(params_state_->get_nv());
    a_constr.segment(0, params_state_->get_nv_pin()) =
        (anitescu_sim_out_->v_next - v_double) / dt_;

    d->xout = a_constr.template cast<Scalar>();

    d->multibody.joint->a = a_constr;
    d->multibody.joint->tau = u;
    costs_->calc(d->costs, x, u); // Use the actual x and u as numerical values
    d->cost = d->costs->cost;
    if (constraints_ != nullptr)
    {
      d->constraints->resize(this, d);
      constraints_->calc(d->constraints, x, u); // Use the actual x and u as numerical values
    }
  }

  template <typename Scalar>
  void DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::calc(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x)
  {
    if (static_cast<std::size_t>(x.size()) != params_state_->get_nx())
    {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(params_state_->get_nx()) + ")");
    }

    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
        x.head(params_state_->get_nq_pin());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
        x.segment(params_state_->get_nq(), params_state_->get_nv_pin());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> p =
        x.segment(params_state_->get_nq_pin(), params_state_->get_np());

    pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);

    costs_->calc(d->costs, x);
    d->cost = d->costs->cost;
    if (constraints_ != nullptr)
    {
      d->constraints->resize(this, d, false);
      constraints_->calc(d->constraints, x);
    }
  }

  template <typename Scalar>
  void DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {

    if (static_cast<std::size_t>(x.size()) != params_state_->get_nx())
    {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(params_state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_)
    {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }

    const std::size_t nv = params_state_->get_nv_pin();
    const std::size_t np = params_state_->get_np();
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
        x.head(params_state_->get_nq_pin());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
        x.segment(params_state_->get_nq(), params_state_->get_nv_pin());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> p =
        x.segment(params_state_->get_nq_pin(), params_state_->get_np());

    Data *d = static_cast<Data *>(data.get());

    actuation_->calcDiff(d->multibody.actuation, x, u);

    // const std::vector<pinocchio::JointIndex> joints =
    //     params_state_->get_param_joints();
    // for (std::size_t jiter = 0; jiter < joints.size(); ++jiter)
    // {
    //   const pinocchio::JointIndex jid = joints[jiter];
    //   const VectorXs p_jid = p.segment(10 * jiter, 10);
    //   pinocchio::LogCholeskyParametersTpl<Scalar> log_chol_jid(p_jid);
    //   const VectorXs pi = log_chol_jid.toDynamicParameters();
    //   const pinocchio::Inertia I =
    //       pinocchio::Inertia::FromDynamicParameters(pi);

    //   pinocchio_.inertias[jid] = I;
    // }

    pinocchio::computeMinverse(pinocchio_, d->pinocchio, q);
    d->M_inv_ = d->pinocchio.Minv;
    d->M_inv_.template triangularView<Eigen::StrictlyLower>() = d->M_inv_.transpose(); // Symmetrize the lowertriangular past

    //--------------------------------------------------------------------------------------------
    // Computing the continuous dynamics derivatives
    //--------------------------------------------------------------------------------------------

    d->Fx.block(0, 0, nv, nv).noalias() =
        anitescu_sim_out_->dv_dq / dt_;
    d->Fx.block(0, nv, nv, np).noalias() =
        anitescu_sim_out_->dv_dtheta / dt_;
    d->Fx.block(0, nv + np, nv, nv).noalias() =
        (anitescu_sim_out_->dv_dv - MatrixXs::Identity(nv, nv)) / dt_;

    d->Fu.topRows(nv).noalias() =
        anitescu_sim_out_->dv_dtau * d->multibody.actuation->dtau_du / dt_;

    d->multibody.joint->da_dx = d->Fx;
    d->multibody.joint->da_du = d->Fu;

    costs_->calcDiff(d->costs, x, u);
    if (constraints_ != nullptr)
    {
      constraints_->calcDiff(d->constraints, x, u);
    }
  }

  template <typename Scalar>
  void DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x)
  {
    if (static_cast<std::size_t>(x.size()) != params_state_->get_nx())
    {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(params_state_->get_nx()) + ")");
    }
    Data *d = static_cast<Data *>(data.get());
    costs_->calcDiff(d->costs, x);
    if (constraints_ != nullptr)
    {
      constraints_->calcDiff(d->constraints, x);
    }
  }

  template <typename Scalar>
  boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::createData()
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  template <typename Scalar>
  bool DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::checkData(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data)
  {
    boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
    if (d != NULL)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename Scalar>
  void DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::print(
      std::ostream &os) const
  {
    os << "DifferentialActionModelContactFwdDynamics {nx=" << params_state_->get_nx()
       << ", ndx=" << params_state_->get_ndx() << ", nu=" << nu_ << "}";
  }

  template <typename Scalar>
  std::size_t DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_ng()
      const
  {
    if (constraints_ != nullptr)
    {
      return constraints_->get_ng();
    }
    else
    {
      return Base::get_ng();
    }
  }

  template <typename Scalar>
  std::size_t DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_nh()
      const
  {
    if (constraints_ != nullptr)
    {
      return constraints_->get_nh();
    }
    else
    {
      return Base::get_nh();
    }
  }

  template <typename Scalar>
  std::size_t DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_ng_T()
      const
  {
    if (constraints_ != nullptr)
    {
      return constraints_->get_ng_T();
    }
    else
    {
      return Base::get_ng_T();
    }
  }

  template <typename Scalar>
  std::size_t DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_nh_T()
      const
  {
    if (constraints_ != nullptr)
    {
      return constraints_->get_nh_T();
    }
    else
    {
      return Base::get_nh_T();
    }
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_g_lb() const
  {
    if (constraints_ != nullptr)
    {
      return constraints_->get_lb();
    }
    else
    {
      return g_lb_;
    }
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_g_ub() const
  {
    if (constraints_ != nullptr)
    {
      return constraints_->get_ub();
    }
    else
    {
      return g_lb_;
    }
  }

  template <typename Scalar>
  pinocchio::ModelTpl<Scalar> &
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_pinocchio() const
  {
    return pinocchio_;
  }

  template <typename Scalar>
  const boost::shared_ptr<ActuationModelAbstractTpl<Scalar>> &
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_actuation() const
  {
    return actuation_;
  }

  template <typename Scalar>
  const boost::shared_ptr<CostModelSumTpl<Scalar>> &
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_costs() const
  {
    return costs_;
  }

  template <typename Scalar>
  const boost::shared_ptr<ConstraintModelManagerTpl<Scalar>> &
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_constraints() const
  {
    return constraints_;
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &
  DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::get_armature() const
  {
    return armature_;
  }

  template <typename Scalar>
  void DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<Scalar>::set_armature(
      const VectorXs &armature)
  {
    if (static_cast<std::size_t>(armature.size()) != params_state_->get_nv())
    {
      throw_pretty("Invalid argument: "
                   << "The armature dimension is wrong (it should be " +
                          std::to_string(params_state_->get_nv()) + ")");
    }
    armature_ = armature;
    with_armature_ = false;
  }

} // namespace crocoddyl
