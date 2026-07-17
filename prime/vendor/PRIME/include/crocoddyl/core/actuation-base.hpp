///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/pointer_cast.hpp>  // dynamic_pointer_cast
#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

// NEW: for StateMultibodyBaseTpl
#include "crocoddyl/multibody/states/multibody_base.hpp"

namespace crocoddyl
{

  template <typename _Scalar>
  class ActuationModelAbstractTpl
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef StateAbstractTpl<Scalar> StateAbstract;
    typedef StateMultibodyBaseTpl<Scalar> StateMultibodyBase;
    typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the actuation model
     *
     * @param[in] state  State description
     * @param[in] nu     Dimension of joint-torque input
     */
    ActuationModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                              const std::size_t nu);

    // IMPORTANT:
    // Do NOT add a second overload with StateMultibodyBase here.
    // It causes overload ambiguity when you pass a derived multibody state
    // (because it can convert to both shared_ptr<StateAbstract> and shared_ptr<StateMultibodyBase>).
    //
    // If you *really* need it, prefer a named static factory or explicit cast at call site.

    virtual ~ActuationModelAbstractTpl();

    virtual void calc(const boost::shared_ptr<ActuationDataAbstract> &data,
                      const Eigen::Ref<const VectorXs> &x,
                      const Eigen::Ref<const VectorXs> &u) = 0;

    void calc(const boost::shared_ptr<ActuationDataAbstract> &data,
              const Eigen::Ref<const VectorXs> &x);

    virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &u) = 0;

    void calcDiff(const boost::shared_ptr<ActuationDataAbstract> &data,
                  const Eigen::Ref<const VectorXs> &x);

    virtual void commands(const boost::shared_ptr<ActuationDataAbstract> &data,
                          const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &tau) = 0;

    virtual void torqueTransform(
        const boost::shared_ptr<ActuationDataAbstract> &data,
        const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u);

    virtual boost::shared_ptr<ActuationDataAbstract> createData();

    std::size_t get_nu() const;

    const boost::shared_ptr<StateAbstract> &get_state() const;

    template <class Scalar2>
    friend std::ostream &operator<<(
        std::ostream &os, const ResidualModelAbstractTpl<Scalar2> &model);

    virtual void print(std::ostream &os) const;

  protected:
    std::size_t nu_;
    boost::shared_ptr<StateAbstract> state_;
  };

  template <typename _Scalar>
  struct ActuationDataAbstractTpl
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef StateAbstractTpl<Scalar> StateAbstract;
    typedef StateMultibodyBaseTpl<Scalar> StateMultibodyBase;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    template <template <typename Scalar2> class Model>
    explicit ActuationDataAbstractTpl(Model<Scalar> *const model)
    {
      const boost::shared_ptr<StateAbstract> &st = model->get_state();

      // Choose nv_pin if we can view this state as multibody, else fallback to nv
      const boost::shared_ptr<StateMultibodyBase> mb =
          boost::dynamic_pointer_cast<StateMultibodyBase>(st);
      const std::size_t nv_eff = mb ? mb->get_nv_pin() : st->get_nv();

      tau.resize(nv_eff);
      u.resize(model->get_nu());
      dtau_dx.resize(nv_eff, st->get_ndx());
      dtau_du.resize(nv_eff, model->get_nu());
      Mtau.resize(model->get_nu(), nv_eff);
      tau_set.assign(nv_eff, true);

      tau.setZero();
      u.setZero();
      dtau_dx.setZero();
      dtau_du.setZero();
      Mtau.setZero();
    }

    virtual ~ActuationDataAbstractTpl() {}

    VectorXs tau;              //!< Generalized torques
    VectorXs u;                //!< Joint torques
    MatrixXs dtau_dx;          //!< Partial derivatives of the actuation model w.r.t. state
    MatrixXs dtau_du;          //!< Partial derivatives of the actuation model w.r.t. input
    MatrixXs Mtau;             //!< Torque transform
    std::vector<bool> tau_set; //!< True for joints that are actuacted
  };

} // namespace crocoddyl

#include "crocoddyl/core/actuation-base.hxx"

#endif // CROCODDYL_CORE_ACTUATION_BASE_HPP_