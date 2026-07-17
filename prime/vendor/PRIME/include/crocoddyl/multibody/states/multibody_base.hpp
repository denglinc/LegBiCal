///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_STATES_MULTIBODY_BASE_HPP_
#define CROCODDYL_MULTIBODY_STATES_MULTIBODY_BASE_HPP_

#include <pinocchio/multibody/model.hpp>

#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl
{
  template <typename Scalar>
  class StateMultibodyBaseTpl : public StateAbstractTpl<Scalar>
  {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using PinocchioModel = pinocchio::ModelTpl<Scalar>;
    using StateAbstractTpl<Scalar>::StateAbstractTpl;

    virtual const boost::shared_ptr<PinocchioModel> &get_pinocchio() const = 0;
    virtual std::size_t get_nq_pin() const = 0;
    virtual std::size_t get_nv_pin() const = 0;

    virtual ~StateMultibodyBaseTpl() = default;
  };
} // namespace crocoddyl


#endif // CROCODDYL_MULTIBODY_STATES_MULTIBODY_BASE_HPP_
