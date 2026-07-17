#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/pointer_cast.hpp>
#include <Eigen/Dense>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/contact_id/anitescu/DifferentiableAnitescuSimulator.hpp"
#include "crocoddyl/contact_id/states/multibody_params.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

namespace stride_prime {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using ActionModel = crocoddyl::ActionModelAbstract;
using ActionData = crocoddyl::ActionDataAbstract;

constexpr int kNv = 7;
constexpr int kNx = 14;

struct Sample {
  double time{};
  Vector q_gt{Vector::Zero(kNv)}, v_gt{Vector::Zero(kNv)};
  Vector u_gt{Vector::Zero(4)};
  Vector q_meas{Vector::Zero(kNv)}, v_meas{Vector::Zero(kNv)};
  Vector u_meas{Vector::Zero(4)};
};

std::vector<double> parseNumbers(const std::string& line) {
  std::vector<double> values;
  std::stringstream stream(line);
  std::string item;
  while (std::getline(stream, item, ',')) values.push_back(std::stod(item));
  return values;
}

std::vector<Sample> loadDataset(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open dataset: " + path);
  std::string line;
  std::getline(input, line);
  std::vector<Sample> samples;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto row = parseNumbers(line);
    if (row.size() != 37)
      throw std::runtime_error("dataset must have 37 columns, got " +
                               std::to_string(row.size()));
    Sample s;
    s.time = row[0];
    for (int j = 0; j < kNv; ++j) {
      s.q_gt[j] = row[1 + j];
      s.v_gt[j] = row[8 + j];
      s.q_meas[j] = row[19 + j];
      s.v_meas[j] = row[26 + j];
    }
    for (int j = 0; j < 4; ++j) {
      s.u_gt[j] = row[15 + j];
      s.u_meas[j] = row[33 + j];
    }
    samples.push_back(s);
  }
  if (samples.size() < 2) throw std::runtime_error("dataset is too short");
  return samples;
}

Vector frostToPinState(const Vector& q, const Vector& v) {
  static constexpr int order[kNv] = {0, 1, 2, 5, 6, 3, 4};
  Vector x(kNx);
  for (int j = 0; j < kNv; ++j) {
    x[j] = q[order[j]];
    x[kNv + j] = v[order[j]];
  }
  return x;
}

Vector pinToFrost(const Vector& pin) {
  static constexpr int order[kNv] = {0, 1, 2, 5, 6, 3, 4};
  Vector frost(kNv);
  for (int j = 0; j < kNv; ++j) frost[order[j]] = pin[j];
  return frost;
}

Vector measuredGeneralizedTorque(const Vector& motor_frost) {
  Vector tau = Vector::Zero(kNv);
  // Pinocchio joint order is left hip/knee followed by right hip/knee.
  tau.segment<2>(3) = 71.2 * motor_frost.segment<2>(2);
  tau.segment<2>(5) = 71.2 * motor_frost.segment<2>(0);
  return tau;
}

Vector defaultMeasurementWeights(double base_q_scale,
                                 double base_velocity_scale) {
  Vector sigma(kNx);
  sigma << 0.002, 0.002, 0.003, 0.006, 0.006, 0.006, 0.006,
           0.020, 0.020, 0.025, 0.040, 0.040, 0.040, 0.040;
  sigma.head<3>() *= base_q_scale;
  sigma.segment<3>(7) *= base_velocity_scale;
  return sigma.cwiseInverse().array().square().matrix();
}

Vector defaultProcessWeights(double base_q_scale,
                             double base_velocity_scale) {
  // Per-step process covariance. Velocities intentionally retain enough
  // freedom to represent impact impulses not explained by noisy measurements.
  Vector sigma(kNx);
  sigma << 0.001, 0.001, 0.0015, 0.012, 0.012, 0.012, 0.012,
           0.030, 0.030, 0.0375, 0.160, 0.160, 0.160, 0.160;
  sigma.head<3>() *= base_q_scale;
  sigma.segment<3>(7) *= base_velocity_scale;
  return sigma.cwiseInverse().array().square().matrix();
}

class ArrivalModel final : public ActionModel {
 public:
  ArrivalModel(const boost::shared_ptr<crocoddyl::StateAbstract>& state,
               const Vector& weight)
      : ActionModel(state, kNx), weight_(weight) {}

  void calc(const boost::shared_ptr<ActionData>& data,
            const Eigen::Ref<const Vector>& x,
            const Eigen::Ref<const Vector>& u) override {
    data->xnext = x + u;
    data->cost = 0.5 * u.dot(weight_.cwiseProduct(u));
  }

  void calcDiff(const boost::shared_ptr<ActionData>& data,
                const Eigen::Ref<const Vector>&,
                const Eigen::Ref<const Vector>& u) override {
    data->Fx.setIdentity();
    data->Fu.setIdentity();
    data->Lx.setZero();
    data->Lu = weight_.cwiseProduct(u);
    data->Lxx.setZero();
    data->Lxu.setZero();
    data->Luu = weight_.asDiagonal();
  }

 private:
  Vector weight_;
};

class ContactProcessModel;

struct ContactProcessData final : public ActionData {
  ContactProcessData(ContactProcessModel* model, const pinocchio::Model& pin_model,
                     const DifferentiableAnitescuSimulator::Options& options);

  DifferentiableAnitescuSimulator simulator;
  StepResult step;
};

class ContactProcessModel final : public ActionModel {
 public:
  ContactProcessModel(
      const boost::shared_ptr<crocoddyl::StateAbstract>& state,
      const boost::shared_ptr<pinocchio::Model>& pin_model,
      const Vector& measurement, const Vector& measured_tau,
      const Vector& measurement_weight, const Vector& process_weight,
      double dt, double kappa, double mu, double newton_tolerance)
      : ActionModel(state, kNx),
        pin_model_(pin_model), measurement_(measurement), tau_(measured_tau),
        measurement_weight_(measurement_weight), process_weight_(process_weight),
        dt_(dt) {
    options_.mu = mu;
    options_.kappa = kappa;
    options_.plane_height = 0.0;
    options_.damping_vector = Vector::Zero(kNv);
    options_.armature_vector = Vector::Zero(kNv);
    options_.armature_vector.tail(4).setConstant(1.683054);
    // The contact velocity is only used to about 1e-6 first-order accuracy by
    // the outer FDDP linearization. Tighter inner solves add line-search work
    // without a measurable estimator benefit on STRIDE impact nodes.
    options_.newton.grad_tol = newton_tolerance;
  }

  void calc(const boost::shared_ptr<ActionData>& base_data,
            const Eigen::Ref<const Vector>& x,
            const Eigen::Ref<const Vector>& w) override {
    auto data = boost::static_pointer_cast<ContactProcessData>(base_data);
    data->step = data->simulator.step(x.head(kNv), x.tail(kNv), tau_, dt_);
    checkContactStep(data->step);
    data->xnext.head(kNv) = x.head(kNv) + dt_ * data->step.v_next;
    data->xnext.tail(kNv) = data->step.v_next;
    data->xnext += w;
    const Vector residual = x - measurement_;
    data->cost = 0.5 * residual.dot(measurement_weight_.cwiseProduct(residual)) +
                 0.5 * w.dot(process_weight_.cwiseProduct(w));
  }

  void calcDiff(const boost::shared_ptr<ActionData>& base_data,
                const Eigen::Ref<const Vector>& x,
                const Eigen::Ref<const Vector>& w) override {
    auto data = boost::static_pointer_cast<ContactProcessData>(base_data);
    data->Fx.setZero();
    data->Fx.topLeftCorner(kNv, kNv).setIdentity();
    data->Fx.topLeftCorner(kNv, kNv).noalias() += dt_ * data->step.dv_dq;
    data->Fx.topRightCorner(kNv, kNv).noalias() = dt_ * data->step.dv_dv;
    data->Fx.bottomLeftCorner(kNv, kNv) = data->step.dv_dq;
    data->Fx.bottomRightCorner(kNv, kNv) = data->step.dv_dv;
    data->Fu.setIdentity();
    data->Lx = measurement_weight_.cwiseProduct(x - measurement_);
    data->Lu = process_weight_.cwiseProduct(w);
    data->Lxx = measurement_weight_.asDiagonal();
    data->Lxu.setZero();
    data->Luu = process_weight_.asDiagonal();
  }

  boost::shared_ptr<ActionData> createData() override {
    return boost::make_shared<ContactProcessData>(this, *pin_model_, options_);
  }

  const Vector& measuredTau() const { return tau_; }

 private:
  static void checkContactStep(const StepResult& result) {
    if (!result.newton_converged || !result.v_next.allFinite() ||
        !result.dv_dq.allFinite() || !result.dv_dv.allFinite() ||
        !(result.min_cone_margin > 0.0)) {
      std::ostringstream message;
      message << "contact Newton failure: converged=" << result.newton_converged
              << " iterations=" << result.newton_iterations
              << " gradient=" << result.newton_gradient_norm
              << " min_cone_margin=" << result.min_cone_margin;
      throw std::runtime_error(message.str());
    }
  }

  boost::shared_ptr<pinocchio::Model> pin_model_;
  Vector measurement_, tau_, measurement_weight_, process_weight_;
  double dt_;
  DifferentiableAnitescuSimulator::Options options_;
};

ContactProcessData::ContactProcessData(
    ContactProcessModel* model, const pinocchio::Model& pin_model,
    const DifferentiableAnitescuSimulator::Options& options)
    : ActionData(static_cast<ActionModel*>(model)),
      simulator(pin_model, {"left_toe", "right_toe"}, options) {}

class TerminalMeasurementModel final : public ActionModel {
 public:
  TerminalMeasurementModel(
      const boost::shared_ptr<crocoddyl::StateAbstract>& state,
      const Vector& measurement, const Vector& weight)
      : ActionModel(state, 0), measurement_(measurement), weight_(weight) {}

  void calc(const boost::shared_ptr<ActionData>& data,
            const Eigen::Ref<const Vector>& x,
            const Eigen::Ref<const Vector>&) override {
    const Vector residual = x - measurement_;
    data->cost = 0.5 * residual.dot(weight_.cwiseProduct(residual));
  }
  void calcDiff(const boost::shared_ptr<ActionData>& data,
                const Eigen::Ref<const Vector>& x,
                const Eigen::Ref<const Vector>&) override {
    data->Lx = weight_.cwiseProduct(x - measurement_);
    data->Lxx = weight_.asDiagonal();
  }

 private:
  Vector measurement_, weight_;
};

std::vector<double> solverAlphas() {
  std::vector<double> values;
  for (int i = 0; i < 11; ++i) values.push_back(std::ldexp(1.0, -i));
  return values;
}

void saveResult(std::ostream& out, const std::vector<Sample>& dataset,
                std::size_t start, const std::vector<Vector>& xs,
                const std::vector<Vector>& ws,
                const std::vector<boost::shared_ptr<ActionData>>& datas,
                const std::vector<Vector>& dynamics_shin,
                const std::vector<Matrix>& dynamics_A_shin,
                const std::vector<Matrix>& dynamics_H_correction) {
  out << "t";
  for (const char* prefix : {"q_est", "v_est", "q_gt", "v_gt", "q_meas", "v_meas"})
    for (int j = 0; j < kNv; ++j) out << ',' << prefix << j;
  for (int j = 0; j < kNx; ++j) out << ",process" << j;
  for (int j = 0; j < 6; ++j) out << ",contact_force" << j;
  for (int j = 0; j < kNx * kNx; ++j) out << ",dynamics_A" << j;
  for (int j = 0; j < kNx; ++j) out << ",dynamics_shin" << j;
  for (int j = 0; j < kNx * kNx; ++j) out << ",dynamics_A_shin" << j;
  for (int j = 0; j < kNx * kNx; ++j) out << ",dynamics_H_correction" << j;
  out << ",newton_iterations,newton_gradient_norm,min_cone_margin\n";
  out << std::setprecision(17);

  // xs[0] is the fixed shooting anchor; xs[1..K] are physical estimates.
  const std::size_t knots = xs.size() - 1;
  for (std::size_t k = 0; k < knots; ++k) {
    const Sample& s = dataset[start + k];
    const Vector q_est = pinToFrost(xs[k + 1].head(kNv));
    const Vector v_est = pinToFrost(xs[k + 1].tail(kNv));
    out << s.time;
    for (const Vector* value : {&q_est, &v_est, &s.q_gt, &s.v_gt,
                                &s.q_meas, &s.v_meas})
      for (int j = 0; j < kNv; ++j) out << ',' << (*value)[j];
    Vector process = Vector::Constant(kNx, std::numeric_limits<double>::quiet_NaN());
    StepResult diagnostics;
    bool has_contact = false;
    if (k + 1 < knots) {
      process = ws[k + 1];
      auto contact_data = boost::dynamic_pointer_cast<ContactProcessData>(datas[k + 1]);
      if (contact_data) {
        diagnostics = contact_data->step;
        has_contact = true;
      }
    }
    for (int j = 0; j < kNx; ++j) out << ',' << process[j];
    for (int j = 0; j < 6; ++j)
      out << ',' << (has_contact && j < diagnostics.force.size()
                         ? diagnostics.force[j]
                         : std::numeric_limits<double>::quiet_NaN());
    Matrix dynamics_A = Matrix::Constant(
        kNx, kNx, std::numeric_limits<double>::quiet_NaN());
    if (has_contact) dynamics_A = datas[k + 1]->Fx;
    for (int row = 0; row < kNx; ++row)
      for (int col = 0; col < kNx; ++col)
        out << ',' << dynamics_A(row, col);
    Vector f_shin = Vector::Constant(kNx, std::numeric_limits<double>::quiet_NaN());
    Matrix A_shin = Matrix::Constant(
        kNx, kNx, std::numeric_limits<double>::quiet_NaN());
    if (k < dynamics_shin.size()) {
      f_shin = dynamics_shin[k];
      A_shin = dynamics_A_shin[k];
    }
    for (int j = 0; j < kNx; ++j) out << ',' << f_shin[j];
    for (int row = 0; row < kNx; ++row)
      for (int col = 0; col < kNx; ++col)
        out << ',' << A_shin(row, col);
    Matrix H_correction = Matrix::Constant(
        kNx, kNx, std::numeric_limits<double>::quiet_NaN());
    if (k < dynamics_H_correction.size())
      H_correction = dynamics_H_correction[k];
    for (int row = 0; row < kNx; ++row)
      for (int col = 0; col < kNx; ++col)
        out << ',' << H_correction(row, col);
    out << ',' << (has_contact ? diagnostics.newton_iterations : -1)
        << ',' << (has_contact ? diagnostics.newton_gradient_norm
                               : std::numeric_limits<double>::quiet_NaN())
        << ',' << (has_contact ? diagnostics.min_cone_margin
                               : std::numeric_limits<double>::quiet_NaN()) << '\n';
  }
}

void setShinLength(pinocchio::Model& model, double length) {
  if (!(length > 0.0)) throw std::runtime_error("shin length must be positive");
  for (const char* name : {"left_toe", "right_toe"}) {
    const pinocchio::FrameIndex id = model.getFrameId(name);
    if (id >= static_cast<pinocchio::FrameIndex>(model.nframes))
      throw std::runtime_error(std::string("missing contact frame: ") + name);
    model.frames[id].placement.translation()[2] = length;
  }
}

struct LocalSensitivityData {
  std::vector<Vector> f_shin;
  std::vector<Matrix> A_shin;
  std::vector<Matrix> hessian_correction;
};

Matrix actionJacobian(const StepResult& step, double dt) {
  Matrix A = Matrix::Zero(kNx, kNx);
  A.topLeftCorner(kNv, kNv).setIdentity();
  A.topLeftCorner(kNv, kNv) += dt * step.dv_dq;
  A.topRightCorner(kNv, kNv) = dt * step.dv_dv;
  A.bottomLeftCorner(kNv, kNv) = step.dv_dq;
  A.bottomRightCorner(kNv, kNv) = step.dv_dv;
  return A;
}

LocalSensitivityData localSensitivityData(
    const pinocchio::Model& nominal_model, double shin_length,
    const std::vector<Sample>& dataset, std::size_t start,
    const std::vector<Vector>& solver_xs, const std::vector<Vector>& solver_us,
    const Vector& process_weight, double dt, double kappa,
    double newton_tolerance) {
  constexpr double epsilon = 2e-5;
  pinocchio::Model plus_model = nominal_model;
  pinocchio::Model minus_model = nominal_model;
  setShinLength(plus_model, shin_length * std::exp(epsilon));
  setShinLength(minus_model, shin_length * std::exp(-epsilon));
  DifferentiableAnitescuSimulator::Options options;
  options.mu = 0.8;
  options.kappa = kappa;
  options.plane_height = 0.0;
  options.damping_vector = Vector::Zero(kNv);
  options.armature_vector = Vector::Zero(kNv);
  options.armature_vector.tail(4).setConstant(1.683054);
  options.newton.grad_tol = newton_tolerance;
  DifferentiableAnitescuSimulator plus(
      plus_model, {"left_toe", "right_toe"}, options);
  DifferentiableAnitescuSimulator minus(
      minus_model, {"left_toe", "right_toe"}, options);
  LocalSensitivityData output;
  output.f_shin.reserve(solver_xs.size() - 2);
  output.A_shin.reserve(solver_xs.size() - 2);
  output.hessian_correction.reserve(solver_xs.size() - 2);
  for (std::size_t k = 0; k + 2 < solver_xs.size(); ++k) {
    const Vector& x = solver_xs[k + 1];
    const Vector tau = measuredGeneralizedTorque(dataset[start + k].u_meas);
    const StepResult p = plus.step(x.head(kNv), x.tail(kNv), tau, dt);
    const StepResult m = minus.step(x.head(kNv), x.tail(kNv), tau, dt);
    if (!p.newton_converged || !m.newton_converged)
      throw std::runtime_error("shin local derivative contact Newton failed");
    Vector fp(kNx), fm(kNx);
    fp << x.head(kNv) + dt * p.v_next, p.v_next;
    fm << x.head(kNv) + dt * m.v_next, m.v_next;
    const Matrix Ap = actionJacobian(p, dt);
    const Matrix Am = actionJacobian(m, dt);
    output.f_shin.push_back((fp - fm) / (2.0 * epsilon));
    output.A_shin.push_back((Ap - Am) / (2.0 * epsilon));

    // Exact KKT Hessian correction, obtained by differentiating the available
    // analytical first-order contact Jacobian. This is the only place where a
    // second derivative is approximated; it never requires another FIE solve.
    Matrix correction(kNx, kNx);
    const Vector weighted_process = process_weight.cwiseProduct(solver_us[k + 1]);
    DifferentiableAnitescuSimulator center(
        nominal_model, {"left_toe", "right_toe"}, options);
    constexpr double state_epsilon = 2e-5;
    for (int j = 0; j < kNx; ++j) {
      Vector xp = x, xm = x;
      xp[j] += state_epsilon;
      xm[j] -= state_epsilon;
      const StepResult sp = center.step(
          xp.head(kNv), xp.tail(kNv), tau, dt);
      const StepResult sm = center.step(
          xm.head(kNv), xm.tail(kNv), tau, dt);
      if (!sp.newton_converged || !sm.newton_converged)
        throw std::runtime_error("second-order local contact derivative failed");
      correction.col(j) =
          (actionJacobian(sp, dt) - actionJacobian(sm, dt)).transpose() *
          weighted_process / (2.0 * state_epsilon);
    }
    output.hessian_correction.push_back(
        0.5 * (correction + correction.transpose()));
  }
  return output;
}

}  // namespace stride_prime

int main(int argc, char** argv) {
  using namespace stride_prime;
  try {
    if (argc < 8 || argc > 13) {
      std::cerr << "usage: prime_fie MODEL DATASET OUTPUT START "
                   "KNOTS MAX_ITER KAPPA [MEAS_BASE_Q_SCALE] "
                   "[MEAS_BASE_V_SCALE] [PROCESS_BASE_Q_SCALE] "
                   "[PROCESS_BASE_V_SCALE] [SHIN_LENGTH_SCALE]\n";
      return 2;
    }
    const std::string model_path = argv[1];
    const std::string dataset_path = argv[2];
    const std::string output_path = argv[3];
    const std::size_t start = std::stoul(argv[4]);
    const std::size_t knots = std::stoul(argv[5]);
    const std::size_t max_iter = std::stoul(argv[6]);
    const double kappa = std::stod(argv[7]);
    const double measurement_q_scale = argc >= 9 ? std::stod(argv[8]) : 1.0;
    const double measurement_v_scale = argc >= 10 ? std::stod(argv[9]) : 1.0;
    const double process_q_scale = argc >= 11 ? std::stod(argv[10]) : 1.0;
    const double process_v_scale = argc >= 12 ? std::stod(argv[11]) : 1.0;
    const double shin_length_scale = argc >= 13 ? std::stod(argv[12]) : 1.0;
    if (!(measurement_q_scale > 0.0) || !(measurement_v_scale > 0.0) ||
        !(process_q_scale > 0.0) || !(process_v_scale > 0.0) ||
        !(shin_length_scale > 0.0))
      throw std::runtime_error("covariance and shin-length scales must be positive");

    const auto dataset = loadDataset(dataset_path);
    const bool stress_scenario =
        dataset_path.find("high_noise") != std::string::npos ||
        dataset_path.find("stress_noise") != std::string::npos;
    const double newton_tolerance = stress_scenario ? 2e-5 : 1e-6;
    if (knots < 2 || start + knots > dataset.size())
      throw std::runtime_error("requested horizon is outside dataset");
    const double dt = dataset[start + 1].time - dataset[start].time;
    if (!(dt > 0.0)) throw std::runtime_error("non-positive sample interval");

    pinocchio::Model model;
    pinocchio::urdf::buildModel(model_path, model);
    constexpr double nominal_shin_length = 0.226;
    const double shin_length = nominal_shin_length * shin_length_scale;
    setShinLength(model, shin_length);
    auto model_ptr = boost::make_shared<pinocchio::Model>(model);
    auto state = boost::make_shared<crocoddyl::StateMultibodyParams>(
        model_ptr, std::vector<pinocchio::JointIndex>{});
    if (state->get_nx() != kNx || state->get_ndx() != kNx)
      throw std::runtime_error("paper FIE currently requires nq=nv=7");

    const Vector measurement_weight = defaultMeasurementWeights(
        measurement_q_scale, measurement_v_scale);
    const Vector process_weight = defaultProcessWeights(
        process_q_scale, process_v_scale);
    std::vector<Vector> measurements;
    measurements.reserve(knots);
    for (std::size_t k = 0; k < knots; ++k)
      measurements.push_back(frostToPinState(dataset[start + k].q_meas,
                                             dataset[start + k].v_meas));

    std::vector<boost::shared_ptr<ActionModel>> running;
    running.reserve(knots);
    running.push_back(boost::make_shared<ArrivalModel>(state, measurement_weight));
    for (std::size_t k = 0; k + 1 < knots; ++k) {
      running.push_back(boost::make_shared<ContactProcessModel>(
          state, model_ptr, measurements[k],
          measuredGeneralizedTorque(dataset[start + k].u_meas),
          measurement_weight, process_weight, dt, kappa, 0.8,
          newton_tolerance));
    }
    auto terminal = boost::make_shared<TerminalMeasurementModel>(
        state, measurements.back(), measurement_weight);
    auto problem = boost::make_shared<crocoddyl::ShootingProblem>(
        measurements.front(), running, terminal);

    std::vector<Vector> xs;
    std::vector<Vector> ws;
    xs.reserve(knots + 1);
    ws.reserve(knots);
    xs.push_back(measurements.front());
    for (const Vector& measurement : measurements) xs.push_back(measurement);
    for (std::size_t k = 0; k < knots; ++k) ws.push_back(Vector::Zero(kNx));

    crocoddyl::SolverFDDP solver(problem);
    solver.set_alphas(solverAlphas());
    const bool converged = solver.solve(xs, ws, max_iter, false, 0.1);
    problem->calc(solver.get_xs(), solver.get_us());
    LocalSensitivityData local_sensitivity;
    if (argc == 13) {
      local_sensitivity = localSensitivityData(
          model, shin_length, dataset, start, solver.get_xs(), solver.get_us(),
          process_weight, dt, kappa, newton_tolerance);
    }
    if (output_path == "-") {
      saveResult(std::cout, dataset, start, solver.get_xs(), solver.get_us(),
                 problem->get_runningDatas(), local_sensitivity.f_shin,
                 local_sensitivity.A_shin,
                 local_sensitivity.hessian_correction);
    } else {
      std::ofstream output(output_path);
      if (!output)
        throw std::runtime_error("cannot write result: " + output_path);
      saveResult(output, dataset, start, solver.get_xs(), solver.get_us(),
                 problem->get_runningDatas(), local_sensitivity.f_shin,
                 local_sensitivity.A_shin,
                 local_sensitivity.hessian_correction);
    }

    const double outer_acceptance_tolerance =
        std::max(1e-5, newton_tolerance);
    const bool accepted = converged ||
        (solver.get_stop() < outer_acceptance_tolerance &&
         solver.get_feas() < 1e-8);
    return accepted ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "prime_fie: " << error.what() << '\n';
    return 1;
  }
}
