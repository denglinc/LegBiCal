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
// interfaces for estimation-oriented contact identification.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_ID_ACTUATIONS_FLOATING_BASE_ESTIMATION_HPP_
#define CROCODDYL_CONTACT_ID_ACTUATIONS_FLOATING_BASE_ESTIMATION_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/states/multibody_base.hpp"

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
  class ActuationModelFloatingBaseEstimationTpl
      : public ActuationModelAbstractTpl<_Scalar>
  {
  public:
    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ActuationModelAbstractTpl<Scalar> Base;
    typedef ActuationDataAbstractTpl<Scalar> Data;
    typedef StateMultibodyBaseTpl<Scalar> StateMultibody;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the floating-base actuation model
     *
     * @param[in] state  State of a multibody system
     * @param[in] nu     Dimension of joint-torque vector
     */
    explicit ActuationModelFloatingBaseEstimationTpl(
        boost::shared_ptr<StateMultibody> state)
        : Base(state,
               state->get_nv_pin()) {};
    virtual ~ActuationModelFloatingBaseEstimationTpl() {};

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
      // if (static_cast<std::size_t>(tau.size()) != state_->get_nv_pin())
      // {
      //   throw_pretty(
      //       "Invalid argument: " << "tau has wrong dimension (it should be " +
      //                                   std::to_string(state_->get_nv_pin()) + ")");
      // }
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

    virtual boost::shared_ptr<Data> createData() override
    {
      boost::shared_ptr<Data> data =
          boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
      typedef StateMultibodyBaseTpl<Scalar> StateMultibody;
      boost::shared_ptr<StateMultibody> state_cast =
          boost::static_pointer_cast<StateMultibody>(state_);

      // Use the base interface (works for StateMultibodyTpl and StateMultibodyParamsTpl)
      const pinocchio::ModelTpl<Scalar> &model = *state_cast->get_pinocchio();

      const std::size_t root_joint_id =
          model.existJointName("root_joint") ? model.getJointId("root_joint") : 0;
      const std::size_t nfb = model.joints[root_joint_id].nv();
      (void)nfb; // keep if you don't use it yet

      // Constant mappings
      data->dtau_du.diagonal().setOnes();
      data->Mtau.setIdentity();

#ifndef NDEBUG
      dtau_du_ = data->dtau_du;
      Mtau_ = data->Mtau;
#endif
      return data;
    }

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

#endif // CROCODDYL_CONTACT_ID_ACTUATIONS_FLOATING_BASE_ESTIMATION_HPP_
