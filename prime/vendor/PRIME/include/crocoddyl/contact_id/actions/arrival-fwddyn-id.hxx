///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
//
// Contact-ID modifications:
//   Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
//   University of Wisconsin-Madison
//
// This file is derived from and extends Crocoddyl action-model interfaces for
// differentiable contact estimation and inertial-parameter identification.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl
{

  template <typename Scalar>
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::ActionModelArrivalFwdDynamicsIDTpl(
      boost::shared_ptr<StateMultibodyParams> state,
      boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs,
      const bool enable_force)
      : Base(state, actuation->get_nu(), costs->get_nr(), 0, 0),
        actuation_(actuation),
        costs_(costs),
        constraints_(nullptr),
        pinocchio_(*state->get_pinocchio().get()),
        enable_force_(enable_force),
        gravity_(state->get_pinocchio()->gravity)
  {
  }

  template <typename Scalar>
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::ActionModelArrivalFwdDynamicsIDTpl(
      boost::shared_ptr<StateMultibodyParams> state,
      boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs,
      boost::shared_ptr<ConstraintModelManager> constraints,
      const bool enable_force)
      : Base(state, actuation->get_nu(), costs->get_nr(), constraints->get_ng(),
             constraints->get_nh(), constraints->get_ng_T(),
             constraints->get_nh_T()),
        actuation_(actuation),
        costs_(costs),
        constraints_(constraints),
        pinocchio_(*state->get_pinocchio().get()),
        enable_force_(enable_force),
        gravity_(state->get_pinocchio()->gravity)
  {
  }

  template <typename Scalar>
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::~ActionModelArrivalFwdDynamicsIDTpl() {}

  template <typename Scalar>
  void ActionModelArrivalFwdDynamicsIDTpl<Scalar>::calc(
      const boost::shared_ptr<ActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    Data *d = static_cast<Data *>(data.get());

    // Computing impulse dynamics and forces
    if (static_cast<std::size_t>(x.size()) != state_->get_nx())
    {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }

    const std::size_t nq = state_->get_nq();
    const std::size_t nv = state_->get_nv();
    boost::shared_ptr<StateMultibodyParams> state =
        boost::static_pointer_cast<StateMultibodyParams>(state_);
    const std::size_t np = state->get_np();
    const std::size_t nv_pin = state->get_nv_pin();

    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
        x.head(nq);
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
        x.tail(nv);
    VectorXs dx = VectorXs::Zero(state_->get_ndx());
    dx.segment(nv_pin, np) = u;
    state_->integrate(x, dx, d->xnext);

    // Computing the cost and constraints
    costs_->calc(d->costs, x, u);
    d->cost = d->costs->cost;
    if (constraints_ != nullptr)
    {
      d->constraints->resize(this, d);
      constraints_->calc(d->constraints, x, u);
    }
  }

  template <typename Scalar>
  void ActionModelArrivalFwdDynamicsIDTpl<Scalar>::calc(
      const boost::shared_ptr<ActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x)
  {
    Data *d = static_cast<Data *>(data.get());

    // Computing the cost and constraints
    costs_->calc(d->costs, x);
    d->cost = d->costs->cost;
    if (constraints_ != nullptr)
    {
      d->constraints->resize(this, d, false);
      constraints_->calc(d->constraints, x);
    }
  }

  template <typename Scalar>
  void ActionModelArrivalFwdDynamicsIDTpl<Scalar>::calcDiff(
      const boost::shared_ptr<ActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    Data *d = static_cast<Data *>(data.get());

    if (static_cast<std::size_t>(x.size()) != state_->get_nx())
    {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }

    const std::size_t nq = state_->get_nq();
    const std::size_t nv = state_->get_nv();
    boost::shared_ptr<StateMultibodyParams> state =
        boost::static_pointer_cast<StateMultibodyParams>(state_);
    const std::size_t np = state->get_np();
    const std::size_t nv_pin = state->get_nv_pin();

    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
        x.head(nq);
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v =
        x.tail(nv);

    // Computing derivatives of impulse dynamics and forces
    // state_->Jintegrate(x, u, d->Fx, d->Fx, first, setto);
    // state_->Jintegrate(x, u, d->Fu, d->Fu, second, setto);

    VectorXs dx = VectorXs::Zero(state_->get_ndx());
    dx.segment(nv_pin, np) = u;

    MatrixXs Fdx = MatrixXs::Zero(state_->get_ndx(), state_->get_ndx());
    state_->Jintegrate(x, dx, d->Fx, d->Fx, first, setto);
    state_->Jintegrate(x, dx, Fdx, Fdx, second, setto);

    d->Fu = Fdx.block(0, nv_pin, state_->get_ndx(), np);

    // Computing the cost derivatives
    if (enable_force_)
    {
      // Force?
    }

    // Computing derivatives of cost and constraints
    costs_->calcDiff(d->costs, x, u);
    d->Lx = d->costs->Lx;
    d->Lu = d->costs->Lu;
    d->Lxx = d->costs->Lxx;
    d->Luu = d->costs->Luu;
    d->Lxu = d->costs->Lxu;

    if (constraints_ != nullptr)
    {
      constraints_->calcDiff(d->constraints, x, u);
    }
  }

  template <typename Scalar>
  void ActionModelArrivalFwdDynamicsIDTpl<Scalar>::calcDiff(
      const boost::shared_ptr<ActionDataAbstract> &data,
      const Eigen::Ref<const VectorXs> &x)
  {
    Data *d = static_cast<Data *>(data.get());

    // Computing derivatives of cost and constraints
    costs_->calcDiff(d->costs, x);
    d->Lx = d->costs->Lx;
    d->Lu = d->costs->Lu;
    d->Lxx = d->costs->Lxx;
    d->Luu = d->costs->Luu;
    d->Lxu = d->costs->Lxu;

    if (constraints_ != nullptr)
    {
      constraints_->calcDiff(d->constraints, x);
    }
  }

  template <typename Scalar>
  boost::shared_ptr<ActionDataAbstractTpl<Scalar>>
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::createData()
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  template <typename Scalar>
  bool ActionModelArrivalFwdDynamicsIDTpl<Scalar>::checkData(
      const boost::shared_ptr<ActionDataAbstract> &data)
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
  void ActionModelArrivalFwdDynamicsIDTpl<Scalar>::quasiStatic(
      const boost::shared_ptr<ActionDataAbstract> &, Eigen::Ref<VectorXs>,
      const Eigen::Ref<const VectorXs> &, const std::size_t, const Scalar)
  {
    // do nothing
  }

  template <typename Scalar>
  std::size_t ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_ng() const
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
  std::size_t ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_nh() const
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
  std::size_t ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_ng_T() const
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
  std::size_t ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_nh_T() const
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
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_g_lb() const
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
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_g_ub() const
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
  void ActionModelArrivalFwdDynamicsIDTpl<Scalar>::print(std::ostream &os) const
  {
    os << "ActionModelImpulseFwdDynamics {nx=" << state_->get_nx()
       << ", ndx=" << state_->get_ndx() << "}";
  }

  template <typename Scalar>
  pinocchio::ModelTpl<Scalar> &
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_pinocchio() const
  {
    return pinocchio_;
  }

  template <typename Scalar>
  const boost::shared_ptr<ActuationModelAbstractTpl<Scalar>> &
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_actuation() const
  {
    return actuation_;
  }

  template <typename Scalar>
  const boost::shared_ptr<CostModelSumTpl<Scalar>> &
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_costs() const
  {
    return costs_;
  }

  template <typename Scalar>
  const boost::shared_ptr<ConstraintModelManagerTpl<Scalar>> &
  ActionModelArrivalFwdDynamicsIDTpl<Scalar>::get_constraints() const
  {
    return constraints_;
  }

} // namespace crocoddyl
