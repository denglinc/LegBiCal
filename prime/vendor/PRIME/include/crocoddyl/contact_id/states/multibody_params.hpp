///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
//
// Contact-ID modifications:
//   Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
//   University of Wisconsin-Madison
//
// This file is derived from and extends Crocoddyl multibody state interfaces
// with inertial-parameter components for contact identification.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_ID_STATES_MULTIBODY_PARAMS_HPP_
#define CROCODDYL_CONTACT_ID_STATES_MULTIBODY_PARAMS_HPP_

#include <pinocchio/multibody/model.hpp>

#include "crocoddyl/multibody/states/multibody_base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl
{

  /**
   * @brief State multibody representation
   *
   * A multibody state is described by the configuration point and its tangential
   * velocity, or in other words, by the generalized position and velocity
   * coordinates of a rigid-body system. For this state, we describe its
   * operators: difference, integrates, transport and their derivatives for any
   * Pinocchio model.
   *
   * For more details about these operators, please read the documentation of the
   * `StateAbstractTpl` class.
   *
   * \sa `diff()`, `integrate()`, `Jdiff()`, `Jintegrate()` and
   * `JintegrateTransport()`
   */
  template <typename _Scalar>
  class StateMultibodyParamsTpl : public StateMultibodyBaseTpl<_Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef StateMultibodyBaseTpl<Scalar> Base;
    typedef pinocchio::ModelTpl<Scalar> PinocchioModel;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the multibody state
     *
     * @param[in] model  Pinocchio model
     */
    explicit StateMultibodyParamsTpl(boost::shared_ptr<PinocchioModel> model, std::vector<pinocchio::JointIndex> Joints);
    StateMultibodyParamsTpl();
    virtual ~StateMultibodyParamsTpl();

    /**
     * @brief Generate a zero state.
     *
     * Note that the zero configuration is computed using `pinocchio::neutral`.
     */
    virtual VectorXs zero() const;

    /**
     * @brief Generate a random state
     *
     * Note that the random configuration is computed using `pinocchio::random`
     * which satisfies the manifold definition (e.g., the quaterion definition)
     */
    virtual VectorXs rand() const;

    virtual void diff(const Eigen::Ref<const VectorXs> &x0,
                      const Eigen::Ref<const VectorXs> &x1,
                      Eigen::Ref<VectorXs> dxout) const;
    virtual void integrate(const Eigen::Ref<const VectorXs> &x,
                           const Eigen::Ref<const VectorXs> &dx,
                           Eigen::Ref<VectorXs> xout) const;
    virtual void Jdiff(const Eigen::Ref<const VectorXs> &,
                       const Eigen::Ref<const VectorXs> &,
                       Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                       const Jcomponent firstsecond = both) const;

    virtual void Jintegrate(const Eigen::Ref<const VectorXs> &x,
                            const Eigen::Ref<const VectorXs> &dx,
                            Eigen::Ref<MatrixXs> Jfirst,
                            Eigen::Ref<MatrixXs> Jsecond,
                            const Jcomponent firstsecond = both,
                            const AssignmentOp = setto) const;
    virtual void JintegrateTransport(const Eigen::Ref<const VectorXs> &x,
                                     const Eigen::Ref<const VectorXs> &dx,
                                     Eigen::Ref<MatrixXs> Jin,
                                     const Jcomponent firstsecond) const;

    /**
     * @brief Return the dimension of the params tuple
     */
    std::size_t get_np() const;

    /**
     * @brief Return the Pinocchio model (i.e., model of the rigid body system)
     */
    const boost::shared_ptr<PinocchioModel> &get_pinocchio() const override; // <-- override
    std::size_t get_nq_pin() const override;                                 // <-- NEW
    std::size_t get_nv_pin() const override;                                 // <-- NEW

    std::vector<pinocchio::JointIndex> get_param_joints() const { return joints_; }

  protected:
    using Base::has_limits_;
    using Base::lb_;
    using Base::ndx_;
    using Base::nq_;
    using Base::nv_;
    using Base::nx_;
    using Base::ub_;

    std::size_t nq_pin_;
    std::size_t nv_pin_;
    std::size_t np_;

  private:
    boost::shared_ptr<PinocchioModel> pinocchio_; //!< Pinocchio model
    VectorXs x0_;                                 //!< Zero state
    std::vector<pinocchio::JointIndex> joints_;   //!< Indices of the parameterized joints
  };

} // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/contact_id/states/multibody_params.hxx"

#endif // CROCODDYL_CONTACT_ID_STATES_MULTIBODY_PARAMS_HPP_
