"""Full-information estimator and KKT derivatives for the process models."""

from casadi import *
import numpy as np
import os
import json
import logging
from casadi import Function
from dataclasses import dataclass

from ..config import FatropConfig


LOGGER = logging.getLogger(__name__)


@dataclass
class EstimatorNlp:
    objective: object
    variables: list
    initial_guess: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]
    constraints: list
    constraint_lowers: list[float]
    constraint_uppers: list[float]
    horizon_points: int
    transition_count: int

    @property
    def variable_vector(self):
        return vertcat(*self.variables)

    @property
    def constraint_vector(self):
        return vertcat(*self.constraints) if self.constraints else SX.zeros(0, 1)


def covariance(v):
    """put v=[dxx,dyy,dzz,dxy,dxz,dyz] into 3x3 symmetric matrix"""
    return vertcat(
        horzcat(v[0], v[3], v[4]),
        horzcat(v[3], v[1], v[5]),
        horzcat(v[4], v[5], v[2]))

class FullInformationEstimator:
    
    def __init__(self, horizon, dt_sample, solver_config: FatropConfig | None = None):
        self.N = horizon
        self.DT = dt_sample
        self.solver_config = solver_config or FatropConfig()
        self._solver = None
        self._problem = None
        self._last_primal = None
        self._last_costate = None
        self._last_theta = None
        self._last_tip = None
        self._last_solution = None

    @property
    def weight_parameter_count(self):
        return self.weight_para.size2()

    def set_state_variable(self, xa):
        self.state = xa
        self.n_state = xa.numel()

    def set_output_variable(self, y):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.output = y
        self.y_fn   = Function('y',[self.state], [self.output], ['x0'], ['yf'])
        self.n_output = self.output.numel()

    def set_control_variable(self, u):
        self.ctrl = u
        self.n_ctrl = u.numel()

    def set_noise_variable(self, eta):
        self.noise = eta
        self.n_noise = eta.numel()

    def set_models(self, models_mhe):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'ctrl'), "Define the control variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        candidate = self.state + self.DT * models_mhe
        quaternion = candidate[9:13]
        quaternion /= sqrt(dot(quaternion, quaternion) + 1e-16)
        self.models_discrete = vertcat(
            candidate[:9], quaternion, candidate[13:]
        )
        self.models_fn = Function('models_mhe', [self.state, self.ctrl, self.noise], [self.models_discrete],
                                  ['s', 'c', 'n'], ['models_f'])

    def set_arrival_cost(self, x_hat):
        assert hasattr(self, 'state'), "Define the state variable first!"
        self.P0        = diag(self.weight_para[0, 0:self.n_state])
        # Define filter priori
        error_a        = self.state - x_hat
        self.cost_a    = 1/2 * mtimes(mtimes(transpose(error_a), self.P0), error_a)
        self.cost_a_fn = Function('cost_a', [self.state, self.weight_para], [self.cost_a], ['s','tp'], ['cost_af'])

    def set_cost_models(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'output'), "Define the output variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"

        # ---------- dimensions ----------
        MEAS_LEN  = 12          # R_q(6) + R_qdot(6)
        NOISE_LEN = 6 + 6 + 3 + 3 + 3 + 3   

        # weights parameter vector
        self.weight_para = SX.sym('t_para', 1, self.n_state + MEAS_LEN + NOISE_LEN)

        self.horizon1 = SX.sym('h1')
        self.index    = SX.sym('ki')

        # measurements and contact
        # y = [yv_foot^B(12); pf_mea(12)]
        self.measurement = SX.sym('y', 24, 1)
        self.contact     = SX.sym('c', 4, 1)

        # G for every step(24x9)= 8x3x9 vertically stack
        self.Gmeas       = SX.sym('G', 24, 9)

        # ---------- unpack measurement weights ----------
        idx = self.n_state
        R_q    = covariance(self.weight_para[0, idx:idx+6]); idx += 6   # 3x3
        R_qdot = covariance(self.weight_para[0, idx:idx+6]); idx += 6   # 3x3

        # outputs from models (body frame)
        y_state = self.output
        vB      = y_state[0:3]
        pfB_FR  = y_state[3:6]
        pfB_FL  = y_state[6:9]
        pfB_RR  = y_state[9:12]
        pfB_RL  = y_state[12:15]

        # measurement vectors
        yv_foot = self.measurement[0:12]    # (12x1)
        pf_mea  = self.measurement[12:24]   # (12x1)

        # residuals per leg
        r_v_FR = vB - yv_foot[0:3]
        r_v_FL = vB - yv_foot[3:6]
        r_v_RR = vB - yv_foot[6:9]
        r_v_RL = vB - yv_foot[9:12]

        r_pf_FR = pfB_FR - pf_mea[0:3]
        r_pf_FL = pfB_FL - pf_mea[3:6]
        r_pf_RR = pfB_RR - pf_mea[6:9]
        r_pf_RL = pfB_RL - pf_mea[9:12]

        # ---------- process weights (for R9's omega block and Q) ----------
        idx = self.n_state + MEAS_LEN
        Qa_blk = covariance(self.weight_para[0, idx:idx+6]); idx += 6
        Qw_blk = covariance(self.weight_para[0, idx:idx+6]); idx += 6   # 3x3 as R9 omega block
        Qa_walk = self.weight_para[0, idx:idx+3]; idx += 3
        Qw_walk = self.weight_para[0, idx:idx+3]; idx += 3
        Qswing  = self.weight_para[0, idx:idx+3]; idx += 3
        Qstance = self.weight_para[0, idx:idx+3]; idx += 3

        # R9 = diag(R_q, R_qdot, Qw_blk)  (9x9)
        R9 = vertcat(
                horzcat(R_q,              SX.zeros(3,3), SX.zeros(3,3)),
                horzcat(SX.zeros(3,3),    R_qdot,       SX.zeros(3,3)),
                horzcat(SX.zeros(3,3),    SX.zeros(3,3), Qw_blk)
            )
        # contact gates
        cFR, cFL, cRR, cRL = self.contact[0], self.contact[1], self.contact[2], self.contact[3]
        
        # ---------- split 8 blocks from G (24x9) ----------
        Gv_FR = self.Gmeas[0:3,   :]
        Gv_FL = self.Gmeas[3:6,   :]
        Gv_RR = self.Gmeas[6:9,   :]
        Gv_RL = self.Gmeas[9:12,  :]
        Gp_FR = self.Gmeas[12:15, :]
        Gp_FL = self.Gmeas[15:18, :]
        Gp_RR = self.Gmeas[18:21, :]
        Gp_RL = self.Gmeas[21:24, :]

        # velocity block
        S_v_FR = mtimes(Gv_FR, solve(R9, Gv_FR.T))
        S_v_FL = mtimes(Gv_FL, solve(R9, Gv_FL.T))
        S_v_RR = mtimes(Gv_RR, solve(R9, Gv_RR.T))
        S_v_RL = mtimes(Gv_RL, solve(R9, Gv_RL.T))

        # position block（Gp only get [J_leg,0,0]）
        S_p_FR = mtimes(Gp_FR, solve(R9, Gp_FR.T))
        S_p_FL = mtimes(Gp_FL, solve(R9, Gp_FL.T))
        S_p_RR = mtimes(Gp_RR, solve(R9, Gp_RR.T))
        S_p_RL = mtimes(Gp_RL, solve(R9, Gp_RL.T))

        # use solve as inv
        J_pf = 0.5*mtimes(r_pf_FR.T, solve(S_p_FR, r_pf_FR)) \
            + 0.5*mtimes(r_pf_FL.T, solve(S_p_FL, r_pf_FL)) \
            + 0.5*mtimes(r_pf_RR.T, solve(S_p_RR, r_pf_RR)) \
            + 0.5*mtimes(r_pf_RL.T, solve(S_p_RL, r_pf_RL))

        J_v  = 0.5*cFR*mtimes(r_v_FR.T, solve(S_v_FR, r_v_FR)) \
            + 0.5*cFL*mtimes(r_v_FL.T, solve(S_v_FL, r_v_FL)) \
            + 0.5*cRR*mtimes(r_v_RR.T, solve(S_v_RR, r_v_RR)) \
            + 0.5*cRL*mtimes(r_v_RL.T, solve(S_v_RL, r_v_RL))

        # ---------- process noise Q ----------
        Q_pf_FR = if_else(cFR >= 0.5, Qstance.T, Qswing.T)
        Q_pf_FL = if_else(cFL >= 0.5, Qstance.T, Qswing.T)
        Q_pf_RR = if_else(cRR >= 0.5, Qstance.T, Qswing.T)
        Q_pf_RL = if_else(cRL >= 0.5, Qstance.T, Qswing.T)

        Q_tail_diag = diag(vertcat(Qa_walk.T, Qw_walk.T, Q_pf_FR, Q_pf_FL, Q_pf_RR, Q_pf_RL))  # 18×18
        Z3_18 = SX.zeros(3,18); Z18_6 = SX.zeros(18,6)

        Q = vertcat(
            horzcat(Qa_blk, SX.zeros(3,3), Z3_18),
            horzcat(SX.zeros(3,3), Qw_blk, Z3_18),
            horzcat(Z18_6, Q_tail_diag)
        )

        self.dJ_running = J_pf + J_v + 0.5*mtimes(mtimes(self.noise.T, Q), self.noise)
        self.dJ_fn = Function('dJ_running',
                            [self.state, self.measurement, self.Gmeas, self.contact, self.noise, self.weight_para, self.horizon1, self.index],
                            [self.dJ_running],
                            ['s','m','G','c','n','tp','h1','ind'], ['dJrunf'])

        self.dJ_T = J_pf + J_v
        self.dJ_T_fn = Function('dJ_T',
                                [self.state, self.measurement, self.Gmeas, self.contact, self.weight_para, self.horizon1, self.index],
                                [self.dJ_T],
                                ['s','m','G','c','tp','h1','ind'], ['dJ_Tf'])

    def formulate(
        self,
        Y,
        ctrl,
        x_hat,
        weight_para,
        time,
        contact_seq,
        G_meas,
    ) -> EstimatorNlp:
        """Build a numerical NLP description without creating a solver."""

        Y, ctrl, contact_seq, G_meas, horizon_points = self._validate_window(
            Y, ctrl, time, contact_seq, G_meas
        )
        self.set_arrival_cost(x_hat)
        measurements = [DM(Y[k, :]).reshape((24, 1)) for k in range(len(Y))]
        g_blocks = [DM(G_meas[k, :, :]) for k in range(len(G_meas))]
        return self._formulate_blocks(
            measurements,
            ctrl,
            x_hat,
            weight_para,
            horizon_points,
            contact_seq,
            g_blocks,
        )

    def prepare(
        self, ctrl, x_hat, time, contact_seq,
        measurement_zero, measurement_tip_jacobian,
        g_zero, g_tip_jacobian,
    ) -> None:
        """Create the parametric Fatrop solver once for a fixed data window."""

        horizon_points = min(int(time) + 1, self.N + 1)
        dummy_y = np.zeros((horizon_points, 24), dtype=float)
        _, ctrl, contact_seq, g_zero, horizon_points = self._validate_window(
            dummy_y, ctrl, time, contact_seq, g_zero
        )
        measurement_zero = np.asarray(measurement_zero, dtype=float)
        measurement_tip_jacobian = np.asarray(measurement_tip_jacobian, dtype=float)
        g_tip_jacobian = np.asarray(g_tip_jacobian, dtype=float)
        self.set_arrival_cost(x_hat)

        theta = SX.sym("theta", self.weight_parameter_count, 1)
        tip = SX.sym("tip", 12, 1)
        measurements = []
        g_blocks = []
        for k in range(horizon_points):
            rows = slice(24 * k, 24 * (k + 1))
            measurements.append(
                DM(measurement_zero[k]).reshape((24, 1))
                + DM(measurement_tip_jacobian[rows, :]) @ tip
            )
            g_vector = DM(g_zero[k].reshape(-1)) + DM(g_tip_jacobian[k]) @ tip
            g_blocks.append(transpose(reshape(g_vector, 9, 24)))
        problem = self._formulate_blocks(
            measurements,
            ctrl,
            x_hat,
            theta.T,
            horizon_points,
            contact_seq,
            g_blocks,
        )
        parameters = vertcat(theta, tip)
        nlp = {
            "f": problem.objective,
            "x": problem.variable_vector,
            "g": problem.constraint_vector,
            "p": parameters,
        }
        self._solver = nlpsol(
            "fie_fatrop", "fatrop", nlp, self._fatrop_options(problem)
        )
        self._problem = problem
        self._last_primal = None
        self._last_costate = None
        self._last_theta = None
        self._last_tip = None
        self._last_solution = None

    def _validate_window(self, Y, ctrl, time, contact_seq, G_meas):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'noise'), "Define the noise variable first!"
        assert hasattr(self, 'models_fn'), "Define the models function first!"
        assert hasattr(self, 'dJ_fn'), "Define the cost models function first!"

        Y = np.asarray(Y, dtype=float)
        ctrl = np.asarray(ctrl, dtype=float)
        contact_seq = np.asarray(contact_seq, dtype=float)
        G_meas = np.asarray(G_meas, dtype=float)

        horizon_points = min(int(time) + 1, self.N + 1)
        if horizon_points < 1:
            raise ValueError("time must define at least one estimator sample")
        if len(Y) < horizon_points:
            raise ValueError(
                f"Y has {len(Y)} samples, but {horizon_points} are required"
            )
        if len(ctrl) < horizon_points or len(contact_seq) < horizon_points:
            raise ValueError("control and contact sequences must match the estimator window")
        if len(G_meas) < horizon_points:
            raise ValueError(
                f"G_meas has {len(G_meas)} samples, but {horizon_points} are required"
            )
        return Y, ctrl, contact_seq, G_meas, horizon_points

    def _formulate_blocks(
        self,
        measurements,
        ctrl,
        x_hat,
        weight_para,
        horizon_points,
        contact_seq,
        g_blocks,
    ) -> EstimatorNlp:
        """Assemble interleaved state/noise stages for Fatrop."""

        transitions = horizon_points - 1
        start = len(measurements) - horizon_points

        variables = []
        initial_guess = []
        lower_bounds = []
        upper_bounds = []
        constraints = []
        constraint_lowers = []
        constraint_uppers = []

        xk = SX.sym("X_0", self.n_state, 1)
        variables.append(xk)
        initial_guess += np.asarray(x_hat, dtype=float).reshape(-1).tolist()
        lower_bounds += self.n_state * [-1e20]
        upper_bounds += self.n_state * [1e20]

        objective = self.cost_a_fn(s=xk, tp=weight_para)["cost_af"]

        for k in range(transitions):
            nk = SX.sym(f"N_{k}", self.n_noise, 1)
            variables.append(nk)
            initial_guess += self.n_noise * [0.0]
            lower_bounds += self.n_noise * [-1e20]
            upper_bounds += self.n_noise * [1e20]

            idx = start + k
            gk = g_blocks[idx]
            objective += self.dJ_fn(
                s=xk,
                m=measurements[idx],
                c=DM(contact_seq[idx, :]).reshape((4, 1)),
                n=nk,
                tp=weight_para,
                G=gk,
                h1=transitions,
                ind=k,
            )["dJrunf"]
            control_k = DM(ctrl[idx, :]).reshape((self.n_ctrl, 1))
            x_next_model = self.models_fn(s=xk, c=control_k, n=nk)["models_f"]

            x_next = SX.sym(f"X_{k + 1}", self.n_state, 1)
            variables.append(x_next)
            initial_guess += np.asarray(x_hat, dtype=float).reshape(-1).tolist()
            lower_bounds += self.n_state * [-1e20]
            upper_bounds += self.n_state * [1e20]

            constraints.append(x_next - x_next_model)
            constraint_lowers += self.n_state * [0.0]
            constraint_uppers += self.n_state * [0.0]
            xk = x_next

        terminal_idx = start + transitions
        objective += self.dJ_T_fn(
            s=xk,
            m=measurements[terminal_idx],
            c=DM(contact_seq[terminal_idx, :]).reshape((4, 1)),
            tp=weight_para,
            G=g_blocks[terminal_idx],
            h1=transitions,
            ind=transitions,
        )["dJ_Tf"]

        return EstimatorNlp(
            objective=objective,
            variables=variables,
            initial_guess=initial_guess,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            constraints=constraints,
            constraint_lowers=constraint_lowers,
            constraint_uppers=constraint_uppers,
            horizon_points=horizon_points,
            transition_count=transitions,
        )

    def solve(self, weight_para, tip_offset):
        if self._solver is None:
            raise RuntimeError("prepare must be called before solve")
        theta = np.asarray(weight_para, dtype=float).reshape(-1)
        tip = np.asarray(tip_offset, dtype=float).reshape(-1)
        problem = self._problem
        if theta.size != self.weight_parameter_count:
            raise ValueError(
                f"weight vector has size {theta.size}; expected "
                f"{self.weight_parameter_count}"
            )
        if tip.size != 12:
            raise ValueError(f"tip vector has size {tip.size}; expected 12")
        if (
            self._last_solution is not None
            and np.array_equal(theta, self._last_theta)
            and np.array_equal(tip, self._last_tip)
        ):
            return self._last_solution

        arguments = {
            "x0": (
                problem.initial_guess
                if self._last_primal is None
                else self._last_primal
            ),
            "p": np.concatenate([theta, tip]),
            "lbx": problem.lower_bounds,
            "ubx": problem.upper_bounds,
            "lbg": problem.constraint_lowers,
            "ubg": problem.constraint_uppers,
        }
        if self._last_costate is not None:
            arguments["lam_g0"] = self._last_costate
        sol = self._solver(**arguments)
        stats = self._solver.stats()
        success = bool(stats.get("success", False))
        if not success and self.solver_config.error_on_fail:
            return_status = stats.get("return_status", "unknown")
            raise RuntimeError(f"Fatrop failed with status: {return_status}")

        w_opt = np.asarray(sol["x"].full(), dtype=float).reshape(-1)
        lam_g = np.asarray(sol["lam_g"].full(), dtype=float).reshape(-1)
        state_traj, noise_traj = self._unpack_solution(w_opt, problem.transition_count)
        costate_traj = np.reshape(lam_g, (-1, self.n_state))
        result = {
            "state_traj_opt": state_traj,
            "noise_traj_opt": noise_traj,
            "costate": costate_traj,
            "costate_fatrop": costate_traj,
            "solver_stats": stats,
        }
        self._last_primal = w_opt
        self._last_costate = lam_g
        self._last_theta = theta.copy()
        self._last_tip = tip.copy()
        self._last_solution = result
        return result

    def _fatrop_options(self, problem: EstimatorNlp) -> dict:
        cfg = self.solver_config
        opts = {
            "expand": cfg.expand,
            "print_time": cfg.print_time,
            "error_on_fail": cfg.error_on_fail,
            "verbose": cfg.verbose,
            "equality": [True] * (problem.transition_count * self.n_state),
            "fatrop": {
                "print_level": 0,
                "tol": cfg.tolerance,
                "max_iter": cfg.max_iterations,
            },
        }
        if cfg.structure_detection == "manual":
            opts.update(
                {
                    "structure_detection": "manual",
                    "N": problem.transition_count,
                    "nx": [self.n_state] * problem.horizon_points,
                    "nu": [self.n_noise] * problem.transition_count + [0],
                    "ng": [0] * problem.horizon_points,
                }
            )
        else:
            opts["structure_detection"] = cfg.structure_detection
        opts.update(cfg.extra_options)
        return opts

    def _unpack_solution(self, w_opt: np.ndarray, transitions: int):
        states = []
        noises = []
        offset = 0
        for k in range(transitions + 1):
            states.append(w_opt[offset : offset + self.n_state])
            offset += self.n_state
            if k < transitions:
                noises.append(w_opt[offset : offset + self.n_noise])
                offset += self.n_noise
        noise_traj = np.vstack(noises) if noises else np.zeros((0, self.n_noise))
        return np.vstack(states), noise_traj
    
    def diffKKT(self):
        assert hasattr(self, 'models_fn'), "Define the models function first!"
        assert hasattr(self, 'output'),  "Define the output variable first!"
        assert hasattr(self, 'dJ_fn'),   "Define the cost models function first!"
        assert hasattr(self, 'dJ_T_fn'), "Define the terminal cost function first!"
        assert hasattr(self, 'n_state') and hasattr(self, 'n_noise') and hasattr(self, 'n_ctrl')

        # Window and dimensions
        H  = self.N             # number of transitions
        nx = self.n_state
        nw = self.n_noise
        nu = self.n_ctrl
        
        MEAS_LEN, NOISE_LEN = 12, 24
        tp = SX.sym('tp', 1, nx + MEAS_LEN + NOISE_LEN)     # weight params
        
        Xhat = SX.sym('Xhat', nx, 1)                         # arrival estimate for x0
        Y    = [SX.sym(f"Y_{k}", 24, 1) for k in range(H+1)] # measurements (24x1)
        C    = [SX.sym(f"C_{k}",  4, 1) for k in range(H+1)] # contacts (4x1)
        U    = [SX.sym(f"U_{k}", nu, 1) for k in range(H)]   # controls (nux1)
        X    = [SX.sym(f"X_{k}", nx, 1) for k in range(H+1)] # state (nxx1)
        W    = [SX.sym(f"W_{k}", nw, 1) for k in range(H)]   # noise (nwx1)
        Lambda = [SX.sym(f"lambda_{k}", nx, 1) for k in range(H)]   # Lagrange multiplier (nxx1)

        # full horizon G as a single vector，then slice back into 24×9
        Gvec = SX.sym('Gvec', (H+1)*24*9, 1)
        # Gvec: ((H+1)*24*9) x 1, reshape into 216*1 as row first order
        def G_slice(k):
            start = k*24*9
            v = Gvec[start:start+24*9]          # 216x1
            return transpose(reshape(v, 9, 24)) # back into-> (24x9)，row first order

        
        J = 0
        g = []
        # L = mtimes(Lambda[0].T, (X[0] - Xhat))
        L = 0        
        L += 1/2 * mtimes(mtimes(transpose(X[0] - Xhat), diag(tp[0, 0:nx])), (X[0] - Xhat))
        J += 1/2 * mtimes(mtimes(transpose(X[0] - Xhat), diag(tp[0, 0:nx])), (X[0] - Xhat))
        # g += [(X[0] - Xhat)]
        for k in range(H):

                # running cost
                Gk = G_slice(k)
                L += self.dJ_fn(s=X[k], m=Y[k], c=C[k], n=W[k], tp=tp, G=Gk, h1=H, ind=k)['dJrunf']
                J += self.dJ_fn(s=X[k], m=Y[k], c=C[k], n=W[k], tp=tp, G=Gk, h1=H, ind=k)['dJrunf']

                # models constraint: X_{k+1} - f(X_k, U_k, N_k)
                Xnext = self.models_fn(s=X[k], c=U[k], n=W[k])['models_f']

                L    += mtimes(Lambda[k].T, (X[k+1] - Xnext))
                g += [(X[k+1] - Xnext)]

        GH = G_slice(H)
        L += self.dJ_T_fn(s=X[H], m=Y[H], c=C[H], tp=tp, G=GH, h1=H, ind=H)['dJ_Tf']
        J += self.dJ_T_fn(s=X[H], m=Y[H], c=C[H], tp=tp, G=GH, h1=H, ind=H)['dJ_Tf']

        Xvec = vertcat(*X)            # shape nx*(H+1) x 1
        Wvec = vertcat(*W)            # shape nw*H x 1 
        Lamvec = vertcat(*Lambda)  # shape nx*(H+1) x 1
        Z_vec = vertcat(Xvec, Wvec, Lamvec)
        
        Y_vec = vertcat(*Y)
        U_vec = vertcat(*U)
        C_vec = vertcat(*C)
        g_vec = vertcat(*g)
        self.KKT = gradient(L, Z_vec) 
        self.dKKT_Z = jacobian(self.KKT, Z_vec) 
        adjoint = SX.sym("adjoint", self.KKT.numel(), 1)
        kkt_tp_vjp = jtimes(self.KKT, tp, adjoint, True)
        kkt_y_vjp = jtimes(self.KKT, Y_vec, adjoint, True)
        kkt_g_vjp = jtimes(self.KKT, Gvec, adjoint, True)

        self.Cost_fn = Function('J',    [Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [J],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['Cost_fn'])
        self.g_fn    = Function('g_vec',[Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [g_vec],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['g_fn'])
        self.KKT_fn  = Function('KKT',  [Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [self.KKT],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['KKT_fn'])
        self.dKKT_Z_fn  = Function('dKKT_Z',[Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec], [self.dKKT_Z],
                                ['s','n','costate','y','u','c','prior','tp','G'], ['dKKT_Z_fn'])
        vjp_inputs = [Xvec, Wvec, Lamvec, Y_vec, U_vec, C_vec, Xhat, tp, Gvec, adjoint]
        vjp_names = ['s','n','costate','y','u','c','prior','tp','G','adjoint']
        self.KKT_tp_vjp_fn = Function(
            'KKT_tp_vjp', vjp_inputs, [kkt_tp_vjp], vjp_names, ['KKT_tp_vjp'])
        self.KKT_Y_vjp_fn = Function(
            'KKT_Y_vjp', vjp_inputs, [kkt_y_vjp], vjp_names, ['KKT_Y_vjp'])
        self.KKT_G_vjp_fn = Function(
            'KKT_G_vjp', vjp_inputs, [kkt_g_vjp], vjp_names, ['KKT_G_vjp'])
    @staticmethod
    def quaternion_to_rotation_matrix(q):
        # q: quaternion [qx, qy, qz, qw]

        norm_q = sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2 + 1e-16)
        qx = q[0] / norm_q
        qy = q[1] / norm_q
        qz = q[2] / norm_q
        qw = q[3] / norm_q

        return vertcat(
            horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)),
            horzcat(2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)),
            horzcat(2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2))
        )


    @staticmethod
    def rotation_matrix_log(R):
        eps = 1e-6
        trace_R = R[0,0] + R[1,1] + R[2,2]
        cos_theta = fmin(1.0 - eps, fmax(-1.0 + eps, (trace_R - 1) / 2))
        theta = acos(cos_theta)
        factor = if_else(theta > eps, theta / (2 * sin(theta)), 0.5 + (3 - trace_R) / 12)
        log_R_matrix =  factor * (R - R.T)
        return vertcat(
            log_R_matrix[2,1],
            log_R_matrix[0,2],
            log_R_matrix[1,0]
        )
    
    def diffquat(self):
            H = self.N  # number of transitions; there are (H+1) states/poses

            # Quaternions for MHE estimate and mocap (symbolic)
            Q  = [SX.sym(f"q_{k}",  4, 1) for k in range(H+1)]   # q_mhe(k)
            Qm = [SX.sym(f"qm_{k}", 4, 1) for k in range(H+1)]   # q_mocap(k)

            # Build the scalar loss L
            L = 0
            for k in range(H+1):  # sum k=0..H
                R_mhe   = FullInformationEstimator.quaternion_to_rotation_matrix(Q[k])
                R_mocap = FullInformationEstimator.quaternion_to_rotation_matrix(Qm[k])
                R_rel   = mtimes(R_mhe, R_mocap.T)                 # R_mhe * R_mocap^T
                w_log   = FullInformationEstimator.rotation_matrix_log(R_rel)           # 3x1
                L       = L + 0.5 * mtimes(w_log.T, w_log)

            # dL/d q_k, stack vertically -> (4*(H+1)) x 1
            grad_blocks = [ gradient(L, Q[k]) for k in range(H+1) ]  # each 4x1
            dLdQ = vertcat(*grad_blocks)

            q_stack  = vertcat(*Q)   # (4*(H+1))x1
            qm_stack = vertcat(*Qm)  # (4*(H+1))x1
            self.dL_dQ_fn = Function('dL_dQ',
                                    [q_stack, qm_stack],
                                    [dLdQ],
                                    ['q', 'qm'],
                                    ['dL_dQ'])
            # L 
            self.L_att_fn = Function('L_att',
                                    [q_stack, qm_stack],
                                    [L],
                                    ['q', 'qm'],
                                    ['L'])
            return self.dL_dQ_fn, self.L_att_fn
    

    # —— diffKKT() / diffquat() —— 
    def save_derivative_bundle(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        # save meta for sanity check
        meta = {
            "N": self.N, "DT": self.DT, "n_state": self.n_state, "n_noise": self.n_noise, "n_ctrl": self.n_ctrl,
            "derivative_version": 5,
            "note": "Change any of these? Rebuild and overwrite the cache."
        }
        with open(os.path.join(cache_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # save every Function
        def _save(name):
            fn = getattr(self, name, None)
            if fn is not None: fn.save(os.path.join(cache_dir, f"{name}.casadi"))
        for name in ["Cost_fn","g_fn","KKT_fn","dKKT_Z_fn","KKT_tp_vjp_fn","KKT_Y_vjp_fn","KKT_G_vjp_fn",
                     "dL_dQ_fn","L_att_fn"]:
            _save(name)

    # —— load saved casadi function —— 
    def load_or_build_derivatives(self, cache_dir):
        try:
            with open(os.path.join(cache_dir, "meta.json"), "r") as f:
                meta = json.load(f)
            assert meta["N"] == self.N and meta["n_state"] == self.n_state \
                and meta["n_noise"] == self.n_noise and meta["n_ctrl"] == self.n_ctrl \
                and abs(float(meta.get("DT", -1.0)) - float(self.DT)) < 1e-15, \
                "Cache dims mismatch; rebuild needed."
            assert meta.get("derivative_version") == 5, "Derivative cache version mismatch."

            def _load(name):
                path = os.path.join(cache_dir, f"{name}.casadi")
                if os.path.exists(path):
                    setattr(self, name, Function.load(path))
            for name in ["Cost_fn","g_fn","KKT_fn","dKKT_Z_fn","KKT_tp_vjp_fn","KKT_Y_vjp_fn","KKT_G_vjp_fn",
                         "dL_dQ_fn","L_att_fn"]:
                _load(name)

            #
            assert all(hasattr(self, name) for name in (
                "KKT_fn", "dKKT_Z_fn", "KKT_tp_vjp_fn",
                "KKT_Y_vjp_fn", "KKT_G_vjp_fn")), "Partial cache; rebuild."
            return True  # successfully reloaded from casadi function file
        except (AssertionError, FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            LOGGER.info("Rebuilding derivative cache: %s", error)
            self.diffKKT()
            self.diffquat()
            self.save_derivative_bundle(cache_dir)
            return False
