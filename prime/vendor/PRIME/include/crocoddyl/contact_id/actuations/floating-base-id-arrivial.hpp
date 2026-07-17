///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
//
// Contact-ID modifications:
//   Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
//   University of Wisconsin-Madison
//
// This file is derived from and extends Crocoddyl floating-base actuation
// interfaces for arrival-model terms in contact identification.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_ID_ACTUATIONS_FLOATING_BASE_ARRIVAL_ID_HPP_
#define CROCODDYL_CONTACT_ID_ACTUATIONS_FLOATING_BASE_ARRIVAL_ID_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/contact_id/states/multibody_params.hpp"

namespace crocoddyl
{

  /**
   * @brief Floating-base actuation model
   *
   * It considers the first joint, defined in the Pinocchio model, as the
   * floating-base joints. Then, this joint (that might have various DoFs) is
   * unactuated.
   *
   * The main computations are carrying out in `calc`, and `calcDiff`, where the
   * former computes actuation signal \f$\mathbf{a}\f$ from a given joint-torque
   * input \f$\mathbf{u}\f$ and state point \f$\mathbf{x}\f$, and the latter
   * computes the Jacobians of the actuation-mapping function. Note that
   * `calcDiff` requires to run `calc` first.
   *
   * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  template <typename _Scalar>
  class ActuationModelFloatingBaseArrivalIDTpl
      : public ActuationModelAbstractTpl<_Scalar>
  {
  public:
    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ActuationModelAbstractTpl<Scalar> Base;
    typedef ActuationDataAbstractTpl<Scalar> Data;
    typedef StateMultibodyParamsTpl<Scalar> StateMultibodyParams;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the floating-base actuation model
     *
     * @param[in] state  State of a multibody system
     * @param[in] nu     Dimension of joint-torque vector
     */
    explicit ActuationModelFloatingBaseArrivalIDTpl(
        boost::shared_ptr<StateMultibodyParams> state)
        : Base(state,
               state->get_np()) {};
    virtual ~ActuationModelFloatingBaseArrivalIDTpl() {};

    /**
     * @brief Compute the floating-base actuation signal from the joint-torque
     * input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     *
     * @param[in] data  Actuation data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Base-disturbance + Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calc(const boost::shared_ptr<Data> &data,
                      const Eigen::Ref<const VectorXs> & /*x*/,
                      const Eigen::Ref<const VectorXs> &u)
    {
      if (static_cast<std::size_t>(u.size()) != nu_)
      {
        throw_pretty(
            "Invalid argument: " << "u has wrong dimension (it should be " +
                                        std::to_string(nu_) + ")");
      }
      data->tau.resize(nu_);
      data->tau.tail(nu_) = u;
    };

    /**
     * @brief Compute the Jacobians of the floating-base actuation function
     *
     * @param[in] data  Actuation data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Joint-torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
#ifndef NDEBUG
    virtual void calcDiff(const boost::shared_ptr<Data> &data,
                          const Eigen::Ref<const VectorXs> & /*x*/,
                          const Eigen::Ref<const VectorXs> & /*u*/)
    {
#else
    virtual void calcDiff(const boost::shared_ptr<Data> &,
                          const Eigen::Ref<const VectorXs> & /*x*/,
                          const Eigen::Ref<const VectorXs> & /*u*/)
    {
#endif
      // The derivatives has constant values which were set in createData.
      assert_pretty(data->dtau_dx.isZero(), "dtau_dx has wrong value");
      assert_pretty(MatrixXs(data->dtau_du).isApprox(dtau_du_),
                    "dtau_du has wrong value");
    };

    virtual void commands(const boost::shared_ptr<Data> &data,
                          const Eigen::Ref<const VectorXs> &,
                          const Eigen::Ref<const VectorXs> &tau)
    {
      boost::shared_ptr<StateMultibodyParamsTpl<Scalar>> params_state =
          boost::static_pointer_cast<StateMultibodyParamsTpl<Scalar>>(state_);
      if (static_cast<std::size_t>(tau.size()) != params_state->get_np())
      {
        throw_pretty(
            "Invalid argument: " << "tau has wrong dimension (it should be " +
                                        std::to_string(params_state->get_np()) + ")");
      }
      data->u.resize(nu_);
      data->u = tau.tail(nu_);
    }

#ifndef NDEBUG
    virtual void torqueTransform(const boost::shared_ptr<Data> &data,
                                 const Eigen::Ref<const VectorXs> &,
                                 const Eigen::Ref<const VectorXs> &)
    {
#else
    virtual void torqueTransform(const boost::shared_ptr<Data> &,
                                 const Eigen::Ref<const VectorXs> &,
                                 const Eigen::Ref<const VectorXs> &)
    {
#endif
      // The torque transform has constant values which were set in createData.
      assert_pretty(MatrixXs(data->Mtau).isApprox(Mtau_), "Mtau has wrong value");
    }

    /**
     * @brief Create the floating-base actuation data
     *
     * @return the actuation data
     */
    virtual boost::shared_ptr<Data> createData()
    {
      boost::shared_ptr<StateMultibodyParamsTpl<Scalar>> state =
          boost::static_pointer_cast<StateMultibodyParamsTpl<Scalar>>(state_);
      boost::shared_ptr<Data> data =
          boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
      const std::size_t root_joint_id =
          state->get_pinocchio()->existJointName("root_joint")
              ? state->get_pinocchio()->getJointId("root_joint")
              : 0;
      data->dtau_du.resize(nu_, nu_);
      data->dtau_du.diagonal().setOnes();
      data->Mtau.resize(nu_, nu_);
      data->Mtau.setIdentity();
#ifndef NDEBUG
      dtau_du_ = data->dtau_du;
      Mtau_ = data->Mtau;
#endif
      return data;
    };

  protected:
    using Base::nu_;
    using Base::state_;

#ifndef NDEBUG
  private:
    MatrixXs dtau_du_;
    MatrixXs Mtau_;
#endif
  };

} // namespace crocoddyl

#endif // CROCODDYL_CONTACT_ID_ACTUATIONS_FLOATING_BASE_ARRIVAL_ID_HPP_
