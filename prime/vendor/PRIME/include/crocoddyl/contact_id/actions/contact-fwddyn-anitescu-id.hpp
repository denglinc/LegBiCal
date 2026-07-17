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

#ifndef CROCODDYL_CONTACT_ID_ACTIONS_CONTACT_FWDDYN_ANITESCU_ID_HPP_
#define CROCODDYL_CONTACT_ID_ACTIONS_CONTACT_FWDDYN_ANITESCU_ID_HPP_

#include <stdexcept>
#include <string>

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/contact_id/anitescu/DifferentiableAnitescuSimulator.hpp"
#include "crocoddyl/contact_id/states/multibody_params.hpp"

namespace crocoddyl
{

    /**
     * @brief Differential action model for contact forward dynamics in multibody
     * systems.
     *
     * This class implements contact forward dynamics given a stack of
     * rigid-contacts described in `ContactModelMultipleTpl`, i.e., \f[
     * \left[\begin{matrix}\dot{\mathbf{v}}
     * \\ -\boldsymbol{\lambda}\end{matrix}\right] = \left[\begin{matrix}\mathbf{M}
     * & \mathbf{J}^{\top}_c \\ {\mathbf{J}_{c}} & \mathbf{0}
     * \end{matrix}\right]^{-1} \left[\begin{matrix}\boldsymbol{\tau}_b
     * \\ -\mathbf{a}_0 \\\end{matrix}\right], \f] where \f$\mathbf{q}\in Q\f$,
     * \f$\mathbf{v}\in\mathbb{R}^{nv}\f$ are the configuration point and
     * generalized velocity (its tangent vector), respectively;
     * \f$\boldsymbol{\tau}_b=\boldsymbol{\tau} -
     * \mathbf{h}(\mathbf{q},\mathbf{v})\f$ is the bias forces that depends on the
     * torque inputs \f$\boldsymbol{\tau}\f$ and the Coriolis effect and gravity
     * field \f$\mathbf{h}(\mathbf{q},\mathbf{v})\f$;
     * \f$\mathbf{J}_c\in\mathbb{R}^{nc\times nv}\f$ is the contact Jacobian
     * expressed in the local frame; and \f$\mathbf{a}_0\in\mathbb{R}^{nc}\f$ is the
     * desired acceleration in the constraint space. To improve stability in the
     * numerical integration, we define PD gains that are similar in spirit to
     * Baumgarte stabilization: \f[ \mathbf{a}_0 = \mathbf{a}_{\lambda(c)} - \alpha
     * \,^oM^{ref}_{\lambda(c)}\ominus\,^oM_{\lambda(c)}
     * - \beta\mathbf{v}_{\lambda(c)}, \f] where \f$\mathbf{v}_{\lambda(c)}\f$,
     * \f$\mathbf{a}_{\lambda(c)}\f$ are the spatial velocity and acceleration at
     * the parent body of the contact \f$\lambda(c)\f$, respectively; \f$\alpha\f$
     * and \f$\beta\f$ are the stabilization gains;
     * \f$\,^oM^{ref}_{\lambda(c)}\ominus\,^oM_{\lambda(c)}\f$ is the
     * \f$\mathbb{SE}(3)\f$ inverse composition between the reference contact
     * placement and the current one.
     *
     * The derivatives of the system acceleration and contact forces are computed
     * efficiently based on the analytical derivatives of Recursive Newton Euler
     * Algorithm (RNEA) as described in \cite mastalli-icra20. Note that the
     * algorithm for computing the RNEA derivatives is described in \cite
     * carpentier-rss18.
     *
     * The stack of cost and constraint functions are implemented in
     * `CostModelSumTpl` and `ConstraintModelAbstractTpl`, respectively. The
     * computation of the contact dynamics and its derivatives are carrying out
     * inside `calc()` and `calcDiff()` functions, respectively. It is also
     * important to remark that `calcDiff()` computes the derivatives using the
     * latest stored values by `calc()`. Thus, we need to run `calc()` first.
     *
     * \sa `DifferentialActionModelAbstractTpl`, `calc()`, `calcDiff()`,
     * `createData()`
     */
    template <typename _Scalar>
    class DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl
        : public DifferentialActionModelAbstractTpl<_Scalar>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef _Scalar Scalar;
        typedef DifferentialActionModelAbstractTpl<Scalar> Base;
        typedef DifferentialActionDataContactFwdDynamicsAnitescuSystemidTpl<Scalar> Data;
        typedef DifferentialActionDataAbstractTpl<Scalar>
            DifferentialActionDataAbstract;
        typedef StateMultibodyParamsTpl<Scalar> StateMultibodyParams;
        typedef CostModelSumTpl<Scalar> CostModelSum;
        typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
        typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
        typedef MathBaseTpl<Scalar> MathBase;
        typedef typename MathBase::VectorXs VectorXs;
        typedef typename MathBase::MatrixXs MatrixXs;
        typedef typename MathBase::Vector3s Vector3s;
        typedef typename MathBase::Matrix3s Matrix3s;

        DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl(
            boost::shared_ptr<StateMultibodyParams> state,
            boost::shared_ptr<ActuationModelAbstract> actuation,
            boost::shared_ptr<CostModelSum> costs,
            // boost::shared_ptr<ConstraintModelManager> constraints = nullptr
            std::vector<std::string> contact_frames,
            const Scalar kappa,
            const Scalar mu,
            const Scalar dt,
            const std::string &force_log_path = "");

        DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl(
            boost::shared_ptr<StateMultibodyParams> state,
            boost::shared_ptr<ActuationModelAbstract> actuation,
            boost::shared_ptr<CostModelSum> costs,
            boost::shared_ptr<ConstraintModelManager> constraints,
            std::vector<std::string> contact_frames,
            const Scalar kappa,
            const Scalar mu,
            const Scalar dt,
            const std::string &force_log_path = "");

        virtual ~DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl();

        /**
         * @brief Compute the system acceleration, and cost value
         *
         * It computes the system acceleration using the contact dynamics.
         *
         * @param[in] data  Contact forward-dynamics data
         * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
         * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
         */
        virtual void calc(
            const boost::shared_ptr<DifferentialActionDataAbstract> &data,
            const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u);

        /**
         * @brief Compute the total cost value for nodes that depends only on the
         * state
         *
         * It updates the total cost and the system acceleration is not updated as it
         * is expected to be zero. Additionally, it does not update the contact
         * forces. This function is used in the terminal nodes of an optimal control
         * problem.
         *
         * @param[in] data  Contact forward-dynamics data
         * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
         */
        virtual void calc(
            const boost::shared_ptr<DifferentialActionDataAbstract> &data,
            const Eigen::Ref<const VectorXs> &x);

        /**
         * @brief Compute the derivatives of the contact dynamics, and cost function
         *
         * @param[in] data  Contact forward-dynamics data
         * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
         * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
         */
        virtual void calcDiff(
            const boost::shared_ptr<DifferentialActionDataAbstract> &data,
            const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u);

        /**
         * @brief Compute the derivatives of the cost functions with respect to the
         * state only
         *
         * It updates the derivatives of the cost function with respect to the state
         * only. Additionally, it does not update the contact forces derivatives. This
         * function is used in the terminal nodes of an optimal control problem.
         *
         * @param[in] data  Contact forward-dynamics data
         * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
         */
        virtual void calcDiff(
            const boost::shared_ptr<DifferentialActionDataAbstract> &data,
            const Eigen::Ref<const VectorXs> &x);

        /**
         * @brief Create the contact forward-dynamics data
         *
         * @return contact forward-dynamics data
         */
        virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

        /**
         * @brief Check that the given data belongs to the contact forward-dynamics
         * data
         */
        virtual bool checkData(
            const boost::shared_ptr<DifferentialActionDataAbstract> &data);

        /**
         * @brief Return the number of inequality constraints
         */
        virtual std::size_t get_ng() const;

        /**
         * @brief Return the number of equality constraints
         */
        virtual std::size_t get_nh() const;

        /**
         * @brief Return the number of equality terminal constraints
         */
        virtual std::size_t get_ng_T() const;

        /**
         * @brief Return the number of equality terminal constraints
         */
        virtual std::size_t get_nh_T() const;

        /**
         * @brief Return the lower bound of the inequality constraints
         */
        virtual const VectorXs &get_g_lb() const;

        /**
         * @brief Return the upper bound of the inequality constraints
         */
        virtual const VectorXs &get_g_ub() const;

        /**
         * @brief Return the actuation model
         */
        const boost::shared_ptr<ActuationModelAbstract> &get_actuation() const;

        /**
         * @brief Return the cost model
         */
        const boost::shared_ptr<CostModelSum> &get_costs() const;

        /**
         * @brief Return the constraint model manager
         */
        const boost::shared_ptr<ConstraintModelManager> &get_constraints() const;

        /**
         * @brief Return the Pinocchio model
         */
        pinocchio::ModelTpl<Scalar> &get_pinocchio() const;

        /**
         * @brief Return the armature vector
         */
        const VectorXs &get_armature() const;

        const boost::shared_ptr<StateMultibodyParamsTpl<Scalar>> &get_state_params() const { return params_state_; }

        /**
         * @brief Modify the armature vector
         */
        void set_armature(const VectorXs &armature);

        /**
         * @brief Print relevant information of the contact forward-dynamics model
         *
         * @param[out] os  Output stream object
         */
        virtual void print(std::ostream &os) const;

    protected:
        using Base::g_lb_;  //!< Lower bound of the inequality constraints
        using Base::g_ub_;  //!< Upper bound of the inequality constraints
        using Base::nu_;    //!< Control dimension
        using Base::state_; //!< Model of the state

    private:
        void init();
        boost::shared_ptr<StateMultibodyParams> params_state_;
        boost::shared_ptr<ActuationModelAbstract> actuation_;   //!< Actuation model
        boost::shared_ptr<CostModelSum> costs_;                 //!< Cost model
        boost::shared_ptr<ConstraintModelManager> constraints_; //!< Constraint model
        pinocchio::ModelTpl<Scalar> &pinocchio_;                //!< Pinocchio model
        bool with_armature_;                                    //!< Indicate if we have defined an armature
        VectorXs armature_;                                     //!< Armature vector
        std::vector<std::string> contact_frames_;               //!< Contact frames
        Scalar kappa_;                                          //!< Slackness smoothing factor
        Scalar mu_;                                             //!< Friction coefficient
        Scalar dt_;                                             //!< Time step
        std::string force_log_path_;                            //!< Optional force CSV log path

        std::unique_ptr<DifferentiableAnitescuSimulator> anitescu_sim_;
        std::unique_ptr<StepResult> anitescu_sim_out_;
    };

    template <typename _Scalar>
    struct DifferentialActionDataContactFwdDynamicsAnitescuSystemidTpl
        : public DifferentialActionDataAbstractTpl<_Scalar>
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef _Scalar Scalar;
        typedef MathBaseTpl<Scalar> MathBase;
        typedef DifferentialActionDataAbstractTpl<Scalar> Base;
        typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
        typedef DataCollectorJointActMultibodyTpl<Scalar>
            DataCollectorJointActMultibody;
        typedef typename MathBase::VectorXs VectorXs;
        typedef typename MathBase::MatrixXs MatrixXs;

        template <template <typename Scalar> class Model>
        explicit DifferentialActionDataContactFwdDynamicsAnitescuSystemidTpl(
            Model<Scalar> *const model)
            : Base(model),
              pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
              multibody(
                  &pinocchio, model->get_actuation()->createData(),
                  boost::make_shared<JointDataAbstract>(
                      model->get_state(), model->get_actuation(), model->get_nu())),
              costs(model->get_costs()->createData(&multibody)),
              M_inv_(model->get_state_params()->get_nv_pin(),
                     model->get_state_params()->get_nv_pin())
        {
            multibody.joint->dtau_du.diagonal().setOnes();
            costs->shareMemory(this);
            if (model->get_constraints() != nullptr)
            {
                constraints = model->get_constraints()->createData(&multibody);
                constraints->shareMemory(this);
            }

            M_inv_.setZero();
        }

        pinocchio::DataTpl<Scalar> pinocchio;
        DataCollectorJointActMultibody multibody;
        boost::shared_ptr<CostDataSumTpl<Scalar>> costs;
        boost::shared_ptr<ConstraintDataManagerTpl<Scalar>> constraints;

        MatrixXs M_inv_;

        using Base::cost;
        using Base::Fu;
        using Base::Fx;
        using Base::Lu;
        using Base::Luu;
        using Base::Lx;
        using Base::Lxu;
        using Base::Lxx;
        using Base::r;
        using Base::xout;
    };

} // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/contact_id/actions/contact-fwddyn-anitescu-id.hxx>

#endif // CROCODDYL_CONTACT_ID_ACTIONS_CONTACT_FWDDYN_ANITESCU_ID_HPP_
