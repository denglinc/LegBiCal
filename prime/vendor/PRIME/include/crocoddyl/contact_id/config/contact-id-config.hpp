///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//
// Contact-ID modifications:
//   Copyright (C) 2026, Jiarong Kang, Legged AI Lab,
//   University of Wisconsin-Madison
//
// This file defines XML configuration structures and parsing helpers for the
// contact-ID extension built on top of Crocoddyl.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CONTACT_ID_CONFIG_CONTACT_ID_CONFIG_HPP_
#define CROCODDYL_CONTACT_ID_CONFIG_CONTACT_ID_CONFIG_HPP_

#include <Eigen/Dense>

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pinocchio/multibody/model.hpp>

#include "crocoddyl/multibody/utils/rapidxml.hpp"

namespace crocoddyl {

struct ContactIDWeights {
  double arrival_alpha;
  double arrival_com;
  double arrival_diag;
  double meas_position;
  double meas_position_z;
  double meas_orientation;
  double meas_joint;
  double meas_linearVel;
  double meas_angularVel;
  double meas_jointVel;
  double meas_params;
  double dyn_position;
  double dyn_orientation;
  double dyn_joint;
  double dyn_params;
  double kappa;
  double mu;
  double arrival_scale_alpha;

  ContactIDWeights()
      : arrival_alpha(0.),
        arrival_com(0.),
        arrival_diag(0.),
        meas_position(0.),
        meas_position_z(0.),
        meas_orientation(0.),
        meas_joint(0.),
        meas_linearVel(0.),
        meas_angularVel(0.),
        meas_jointVel(0.),
        meas_params(0.),
        dyn_position(0.),
        dyn_orientation(0.),
        dyn_joint(0.),
        dyn_params(0.),
        kappa(100.),
        mu(0.5),
        arrival_scale_alpha(10.) {}
};

struct ContactIDSolverConfig {
  std::size_t horizon;
  std::size_t down_sample;
  std::size_t start_idx;
  std::size_t max_iter;
  std::size_t n_thread;
  double interval;
  double alpha0;
  bool callbacks;
  bool dry_run;

  ContactIDSolverConfig()
      : horizon(0),
        down_sample(1),
        start_idx(0),
        max_iter(0),
        n_thread(1),
        interval(0.),
        alpha0(1.),
        callbacks(false),
        dry_run(false) {}
};

struct ContactIDContinuationStage {
  double kappa;
  std::size_t max_iter;
  bool has_max_iter;

  ContactIDContinuationStage()
      : kappa(0.), max_iter(0), has_max_iter(false) {}
};

struct ContactIDContinuationConfig {
  bool enabled;
  std::vector<ContactIDContinuationStage> stages;

  ContactIDContinuationConfig() : enabled(false) {}
};

struct ContactIDRobotConfig {
  std::string urdf;
  std::string srdf;
  std::string package_dir;
  std::string reference_configuration;

  ContactIDRobotConfig() : reference_configuration("standing") {}
};

struct ContactIDDataConfig {
  std::string q_csv;
  std::string v_csv;
  std::string u_csv;
  std::string torso_base_frame;
  std::vector<int> joint_order;
  bool q_has_time_column;
  bool v_has_time_column;
  bool u_has_time_column;
  bool normalize_base_quaternion;
  bool shift_base_to_ground;
  bool reuse_initial_ground_height;
  bool average_controls;

  ContactIDDataConfig()
      : q_has_time_column(false),
        v_has_time_column(false),
        u_has_time_column(false),
        normalize_base_quaternion(true),
        shift_base_to_ground(false),
        reuse_initial_ground_height(false),
        average_controls(true) {}
};

struct ContactIDOutputConfig {
  std::string directory;
  std::string xs_log;
  std::string us_log;
  std::string xs_results;
  std::string us_results;
  std::string u0_results;
  std::string inertia_report;
  std::string xs_rollout;
  std::string force_log;
  bool log_initial_guess;
  bool rollout;

  ContactIDOutputConfig()
      : directory("."),
        xs_log("xs_log.csv"),
        us_log("us_log.csv"),
        xs_results("xs_results_fddp.csv"),
        us_results("us_results_fddp.csv"),
        u0_results("u0_results_fddp.csv"),
        inertia_report("inertia_identification.txt"),
        xs_rollout("xs_rollout.csv"),
        force_log("f_rollout.csv"),
        log_initial_guess(false),
        rollout(true) {}
};

struct ContactIDXMLConfig {
  ContactIDRobotConfig robot;
  std::vector<std::string> contact_frames;
  std::vector<std::string> identified_joint_names;
  std::vector<pinocchio::JointIndex> identified_joint_indices;
  std::vector<Eigen::VectorXd> parameter_offsets;
  ContactIDDataConfig data;
  ContactIDSolverConfig solver;
  ContactIDWeights weights;
  ContactIDContinuationConfig continuation;
  ContactIDOutputConfig outputs;
  std::string xml_dir;
};

namespace contact_id_xml {

inline std::string dirname(const std::string& path) {
  const std::string::size_type pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0) {
    return path.substr(0, 1);
  }
  return path.substr(0, pos);
}

inline bool is_absolute_path(const std::string& path) {
  return !path.empty() && (path[0] == '/' || path[0] == '\\' ||
                          (path.size() > 1 && path[1] == ':'));
}

inline std::string join_path(const std::string& lhs, const std::string& rhs) {
  if (rhs.empty() || is_absolute_path(rhs)) {
    return rhs;
  }
  if (lhs.empty() || lhs == ".") {
    return rhs;
  }
  const char last = lhs[lhs.size() - 1];
  if (last == '/' || last == '\\') {
    return lhs + rhs;
  }
  return lhs + "/" + rhs;
}

struct PathResolver {
  explicit PathResolver(const std::string& xml_dir_in) : xml_dir(xml_dir_in) {
    add("xml_dir", xml_dir);
  }

  void add(const std::string& name, const std::string& raw_value) {
    if (name.empty()) {
      return;
    }
    std::string value = expand(raw_value);
    if (!value.empty() && !is_absolute_path(value)) {
      value = join_path(xml_dir, value);
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == name) {
        values[i] = value;
        return;
      }
    }
    names.push_back(name);
    values.push_back(value);
  }

  std::string resolve(const std::string& path) const {
    if (path.empty()) {
      return "";
    }
    return join_path(xml_dir, expand(path));
  }

  std::string expand(const std::string& text) const {
    std::string out;
    for (std::size_t i = 0; i < text.size();) {
      if (text[i] == '$' && i + 1 < text.size() && text[i + 1] == '{') {
        const std::size_t end = text.find('}', i + 2);
        if (end == std::string::npos) {
          throw std::runtime_error("Unclosed XML path variable in: " + text);
        }
        const std::string key = text.substr(i + 2, end - (i + 2));
        bool found = false;
        for (std::size_t j = 0; j < names.size(); ++j) {
          if (names[j] == key) {
            out += values[j];
            found = true;
            break;
          }
        }
        if (!found) {
          throw std::runtime_error("Unknown XML path variable: " + key);
        }
        i = end + 1;
      } else {
        out.push_back(text[i]);
        ++i;
      }
    }
    return out;
  }

  std::string xml_dir;
  std::vector<std::string> names;
  std::vector<std::string> values;
};

inline std::string attr(rapidxml::xml_node<>* node, const char* name,
                        const std::string& defval = "") {
  if (!node) {
    return defval;
  }
  rapidxml::xml_attribute<>* a = node->first_attribute(name);
  return a ? std::string(a->value()) : defval;
}

inline double attr_double(rapidxml::xml_node<>* node, const char* name,
                          double defval = 0.) {
  const std::string value = attr(node, name, "");
  if (value.empty()) {
    return defval;
  }
  return std::stod(value);
}

inline std::size_t attr_size(rapidxml::xml_node<>* node, const char* name,
                             std::size_t defval = 0) {
  const std::string value = attr(node, name, "");
  if (value.empty()) {
    return defval;
  }
  return static_cast<std::size_t>(std::stoul(value));
}

inline bool attr_bool(rapidxml::xml_node<>* node, const char* name,
                      bool defval = false) {
  std::string value = attr(node, name, "");
  if (value.empty()) {
    return defval;
  }
  for (std::size_t i = 0; i < value.size(); ++i) {
    value[i] = static_cast<char>(std::tolower(value[i]));
  }
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

inline double child_value(rapidxml::xml_node<>* parent, const char* child,
                          double defval = 0.) {
  if (!parent) {
    return defval;
  }
  return attr_double(parent->first_node(child), "value", defval);
}

inline Eigen::VectorXd parse_vector(const std::string& text) {
  std::stringstream ss(text);
  std::vector<double> values;
  std::string token;
  while (std::getline(ss, token, ',')) {
    std::stringstream token_stream(token);
    double value = 0.;
    token_stream >> value;
    if (!token_stream.fail()) {
      values.push_back(value);
    }
  }
  Eigen::VectorXd out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = values[i];
  }
  return out;
}

inline std::vector<int> parse_index_vector(const std::string& text) {
  std::stringstream ss(text);
  std::vector<int> values;
  std::string token;
  while (std::getline(ss, token, ',')) {
    std::stringstream token_stream(token);
    int value = 0;
    token_stream >> value;
    if (!token_stream.fail()) {
      values.push_back(value);
    }
  }
  return values;
}

inline std::vector<char> read_xml_buffer(const std::string& xml_path) {
  std::ifstream ifs(xml_path.c_str(), std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open XML config: " + xml_path);
  }
  std::vector<char> buffer((std::istreambuf_iterator<char>(ifs)),
                           std::istreambuf_iterator<char>());
  buffer.push_back('\0');
  return buffer;
}

inline void load_legacy_weights(rapidxml::xml_node<>* root,
                                ContactIDWeights& weights) {
  rapidxml::xml_node<>* xml_noise = root ? root->first_node("noise") : NULL;
  const double legacy_arrival = child_value(xml_noise, "arrival", 0.);
  weights.arrival_alpha = child_value(xml_noise, "arrival_alpha", legacy_arrival);
  weights.arrival_com = child_value(xml_noise, "arrival_com", legacy_arrival);
  weights.arrival_diag = child_value(xml_noise, "arrival_diag", legacy_arrival);
  weights.meas_position = child_value(xml_noise, "meas_position", 0.);
  weights.meas_position_z = child_value(xml_noise, "meas_position_z", 0.);
  weights.meas_orientation = child_value(xml_noise, "meas_orientation", 0.);
  weights.meas_joint = child_value(xml_noise, "meas_joint", 0.);
  weights.meas_linearVel = child_value(xml_noise, "meas_linearVel", 0.);
  weights.meas_angularVel = child_value(xml_noise, "meas_angularVel", 0.);
  weights.meas_jointVel = child_value(xml_noise, "meas_jointVel", 0.);
  weights.meas_params = child_value(xml_noise, "meas_params", 0.);
  weights.dyn_position = child_value(xml_noise, "dyn_position", 0.);
  weights.dyn_orientation = child_value(xml_noise, "dyn_orientation", 0.);
  weights.dyn_joint = child_value(xml_noise, "dyn_joint", 0.);
  weights.dyn_params = child_value(xml_noise, "dyn_params", 0.);
  weights.kappa = child_value(xml_noise, "kappa", weights.kappa);
  weights.mu = child_value(xml_noise, "mu", weights.mu);
  weights.arrival_scale_alpha =
      child_value(xml_noise, "arrival_scale_alpha", weights.arrival_scale_alpha);
}

inline ContactIDWeights load_legacy_weights_file(const std::string& xml_path) {
  std::vector<char> buffer = read_xml_buffer(xml_path);
  rapidxml::xml_document<> doc;
  doc.parse<0>(&buffer[0]);
  ContactIDWeights weights;
  load_legacy_weights(doc.first_node("config"), weights);
  return weights;
}

inline ContactIDXMLConfig load_config(const std::string& xml_path) {
  std::vector<char> buffer = read_xml_buffer(xml_path);
  rapidxml::xml_document<> doc;
  doc.parse<0>(&buffer[0]);
  rapidxml::xml_node<>* root = doc.first_node("config");
  if (!root) {
    throw std::runtime_error("XML config must contain a <config> root.");
  }

  ContactIDXMLConfig cfg;
  cfg.xml_dir = dirname(xml_path);
  PathResolver paths(cfg.xml_dir);
  rapidxml::xml_node<>* path_node = root->first_node("paths");
  for (rapidxml::xml_attribute<>* a =
           path_node ? path_node->first_attribute() : NULL;
       a; a = a->next_attribute()) {
    paths.add(a->name(), a->value());
  }

  rapidxml::xml_node<>* robot = root->first_node("robot");
  cfg.robot.urdf = paths.resolve(attr(robot, "urdf"));
  cfg.robot.srdf = paths.resolve(attr(robot, "srdf"));
  cfg.robot.package_dir = paths.resolve(attr(robot, "package_dir"));
  cfg.robot.reference_configuration =
      attr(robot, "reference_configuration", cfg.robot.reference_configuration);

  rapidxml::xml_node<>* contacts = root->first_node("contacts");
  for (rapidxml::xml_node<>* frame = contacts ? contacts->first_node("frame") : NULL;
       frame; frame = frame->next_sibling("frame")) {
    const std::string name = attr(frame, "name");
    if (!name.empty()) {
      cfg.contact_frames.push_back(name);
    }
  }

  rapidxml::xml_node<>* identification = root->first_node("identification");
  for (rapidxml::xml_node<>* link =
           identification ? identification->first_node("link") : NULL;
       link; link = link->next_sibling("link")) {
    const std::string name = attr(link, "name");
    const std::string index = attr(link, "index");
    if (!name.empty()) {
      cfg.identified_joint_names.push_back(name);
    } else if (!index.empty()) {
      cfg.identified_joint_indices.push_back(
          static_cast<pinocchio::JointIndex>(std::stoul(index)));
    }
    const std::string offset = attr(link, "offset");
    if (!offset.empty()) {
      cfg.parameter_offsets.push_back(parse_vector(offset));
    } else {
      cfg.parameter_offsets.push_back(Eigen::VectorXd());
    }
  }

  rapidxml::xml_node<>* data = root->first_node("data");
  cfg.data.q_csv = paths.resolve(attr(data, "q"));
  cfg.data.v_csv = paths.resolve(attr(data, "v"));
  cfg.data.u_csv = paths.resolve(attr(data, "u"));
  cfg.data.torso_base_frame = attr(data, "torso_base_frame");
  cfg.data.joint_order = parse_index_vector(attr(data, "joint_order"));
  cfg.data.q_has_time_column = attr_bool(data, "q_has_time_column", false);
  cfg.data.v_has_time_column = attr_bool(data, "v_has_time_column", false);
  cfg.data.u_has_time_column = attr_bool(data, "u_has_time_column", false);
  cfg.data.normalize_base_quaternion =
      attr_bool(data, "normalize_base_quaternion", true);
  cfg.data.shift_base_to_ground = attr_bool(data, "shift_base_to_ground", false);
  cfg.data.reuse_initial_ground_height =
      attr_bool(data, "reuse_initial_ground_height", false);
  cfg.data.average_controls = attr_bool(data, "average_controls", true);

  rapidxml::xml_node<>* solver = root->first_node("solver");
  cfg.solver.horizon = attr_size(solver, "horizon", cfg.solver.horizon);
  cfg.solver.down_sample =
      attr_size(solver, "down_sample", cfg.solver.down_sample);
  cfg.solver.start_idx = attr_size(solver, "start_idx", cfg.solver.start_idx);
  cfg.solver.max_iter = attr_size(solver, "max_iter", cfg.solver.max_iter);
  cfg.solver.n_thread = attr_size(solver, "n_thread", cfg.solver.n_thread);
  cfg.solver.interval = attr_double(solver, "interval", cfg.solver.interval);
  cfg.solver.alpha0 = attr_double(solver, "alpha0", cfg.solver.alpha0);
  cfg.solver.callbacks = attr_bool(solver, "callbacks", cfg.solver.callbacks);
  cfg.solver.dry_run = attr_bool(solver, "dry_run", cfg.solver.dry_run);

  rapidxml::xml_node<>* weights = root->first_node("weights");
  if (weights) {
    cfg.weights.arrival_alpha =
        attr_double(weights, "arrival_alpha", cfg.weights.arrival_alpha);
    cfg.weights.arrival_com =
        attr_double(weights, "arrival_com", cfg.weights.arrival_com);
    cfg.weights.arrival_diag =
        attr_double(weights, "arrival_diag", cfg.weights.arrival_diag);
    cfg.weights.meas_position =
        attr_double(weights, "meas_position", cfg.weights.meas_position);
    cfg.weights.meas_position_z =
        attr_double(weights, "meas_position_z", cfg.weights.meas_position_z);
    cfg.weights.meas_orientation =
        attr_double(weights, "meas_orientation", cfg.weights.meas_orientation);
    cfg.weights.meas_joint =
        attr_double(weights, "meas_joint", cfg.weights.meas_joint);
    cfg.weights.meas_linearVel =
        attr_double(weights, "meas_linearVel", cfg.weights.meas_linearVel);
    cfg.weights.meas_angularVel =
        attr_double(weights, "meas_angularVel", cfg.weights.meas_angularVel);
    cfg.weights.meas_jointVel =
        attr_double(weights, "meas_jointVel", cfg.weights.meas_jointVel);
    cfg.weights.meas_params =
        attr_double(weights, "meas_params", cfg.weights.meas_params);
    cfg.weights.dyn_position =
        attr_double(weights, "dyn_position", cfg.weights.dyn_position);
    cfg.weights.dyn_orientation =
        attr_double(weights, "dyn_orientation", cfg.weights.dyn_orientation);
    cfg.weights.dyn_joint = attr_double(weights, "dyn_joint", cfg.weights.dyn_joint);
    cfg.weights.dyn_params =
        attr_double(weights, "dyn_params", cfg.weights.dyn_params);
    cfg.weights.kappa = attr_double(weights, "kappa", cfg.weights.kappa);
    cfg.weights.mu = attr_double(weights, "mu", cfg.weights.mu);
    cfg.weights.arrival_scale_alpha = attr_double(
        weights, "arrival_scale_alpha", cfg.weights.arrival_scale_alpha);
  } else {
    load_legacy_weights(root, cfg.weights);
  }

  rapidxml::xml_node<>* continuation = root->first_node("continuation");
  if (continuation) {
    cfg.continuation.enabled =
        attr_bool(continuation, "enabled", cfg.continuation.enabled);
    for (rapidxml::xml_node<>* stage = continuation->first_node("stage"); stage;
         stage = stage->next_sibling("stage")) {
      const std::string kappa_text = attr(stage, "kappa");
      if (kappa_text.empty()) {
        throw std::runtime_error(
            "Each <continuation><stage> must define a kappa attribute.");
      }
      ContactIDContinuationStage parsed_stage;
      parsed_stage.kappa = std::stod(kappa_text);
      if (parsed_stage.kappa <= 0.) {
        throw std::runtime_error("Continuation stage kappa must be positive.");
      }
      const std::string max_iter_text = attr(stage, "max_iter");
      if (!max_iter_text.empty()) {
        parsed_stage.max_iter = static_cast<std::size_t>(
            std::stoul(max_iter_text));
        parsed_stage.has_max_iter = true;
      }
      cfg.continuation.stages.push_back(parsed_stage);
    }
    if (cfg.continuation.enabled && cfg.continuation.stages.empty()) {
      throw std::runtime_error(
          "Continuation is enabled but no <stage> entries were provided.");
    }
  }

  rapidxml::xml_node<>* outputs = root->first_node("outputs");
  cfg.outputs.directory =
      paths.resolve(attr(outputs, "directory", cfg.outputs.directory));
  cfg.outputs.xs_log = attr(outputs, "xs_log", cfg.outputs.xs_log);
  cfg.outputs.us_log = attr(outputs, "us_log", cfg.outputs.us_log);
  cfg.outputs.xs_results = attr(outputs, "xs_results", cfg.outputs.xs_results);
  cfg.outputs.us_results = attr(outputs, "us_results", cfg.outputs.us_results);
  cfg.outputs.u0_results = attr(outputs, "u0_results", cfg.outputs.u0_results);
  cfg.outputs.inertia_report =
      attr(outputs, "inertia_report", cfg.outputs.inertia_report);
  cfg.outputs.xs_rollout = attr(outputs, "xs_rollout", cfg.outputs.xs_rollout);
  cfg.outputs.force_log = attr(outputs, "force_log", cfg.outputs.force_log);
  cfg.outputs.log_initial_guess =
      attr_bool(outputs, "log_initial_guess", cfg.outputs.log_initial_guess);
  cfg.outputs.rollout = attr_bool(outputs, "rollout", cfg.outputs.rollout);

  return cfg;
}

}  // namespace contact_id_xml
}  // namespace crocoddyl

#endif  // CROCODDYL_CONTACT_ID_CONFIG_CONTACT_ID_CONFIG_HPP_
