"""Process and output models used by the full-information estimator."""

from casadi import *
import numpy as np


class Models:
    def __init__(self, dt_sample):
        self.p = SX.sym("p", 3, 1)
        self.v = SX.sym("v", 3, 1)
        self.qx, self.qy, self.qz, self.qw = (
            SX.sym("qx"),
            SX.sym("qy"),
            SX.sym("qz"),
            SX.sym("qw"),
        )
        self.q = vertcat(self.qx, self.qy, self.qz, self.qw)
        self.pf = SX.sym("pf", 12, 1)

        self.omegax, self.omegay, self.omegaz = (
            SX.sym("omegax"),
            SX.sym("omegay"),
            SX.sym("omegaz"),
        )
        self.omega = vertcat(self.omegax, self.omegay, self.omegaz)
        self.omega_walk = SX.sym("omega_walk", 3, 1)
        self.a = SX.sym("a", 3, 1)
        self.a_walk = SX.sym("a_walk", 3, 1)
        self.u = vertcat(self.a, self.omega)

        self.wa = SX.sym("wa", 3, 1)
        self.wa_walk = SX.sym("wa_walk", 3, 1)
        self.womega = SX.sym("womega", 3, 1)
        self.womega_walk = SX.sym("womega_walk", 3, 1)
        self.wpf = SX.sym("wpf", 12, 1)
        self.w = vertcat(
            self.wa, self.womega, self.wa_walk, self.womega_walk, self.wpf
        )

        self.dt = dt_sample
        self.ex = vertcat(1, 0, 0)
        self.ey = vertcat(0, 1, 0)
        self.ez = vertcat(0, 0, 1)
        self.g = 9.81

    def noisy_rate(self, x, u, noise):
        p = vertcat(x[0, 0], x[1, 0], x[2, 0])
        v = vertcat(x[3, 0], x[4, 0], x[5, 0])
        a_walk = vertcat(x[6, 0], x[7, 0], x[8, 0])
        q = vertcat(x[9, 0], x[10, 0], x[11, 0], x[12, 0])
        omega_walk = vertcat(x[13, 0], x[14, 0], x[15, 0])
        pf = vertcat(x[16:28, 0])

        wa = vertcat(noise[0, 0], noise[1, 0], noise[2, 0])
        womega = vertcat(noise[3, 0], noise[4, 0], noise[5, 0])
        wa_walk = vertcat(noise[6, 0], noise[7, 0], noise[8, 0])
        womega_walk = vertcat(noise[9, 0], noise[10, 0], noise[11, 0])
        wpf = vertcat(noise[12:24, 0])

        rotation = self.quaternion_to_rotation_matrix(q)
        a = vertcat(u[0, 0], u[1, 0], u[2, 0]) + wa + a_walk
        omega = vertcat(u[3, 0], u[4, 0], u[5, 0]) + womega + omega_walk

        dp = v
        dv = -self.g * self.ez + mtimes(rotation, a)
        da = wa_walk
        dq = 0.5 * self.quat_multiply_elementwise(q, vertcat(omega, DM(0.0)))
        domega = womega_walk
        dpf = wpf
        return vertcat(dp, dv, da, dq, domega, dpf)

    def build_models(self):
        self.xa = vertcat(
            self.p, self.v, self.a_walk, self.q, self.omega_walk, self.pf
        )
        self.x = vertcat(self.p, self.v, self.q, self.pf)

        rotation = self.quaternion_to_rotation_matrix(self.q)
        v_body = mtimes(rotation.T, self.v)
        pf_fr_w = self.pf[0:3]
        pf_fl_w = self.pf[3:6]
        pf_rr_w = self.pf[6:9]
        pf_rl_w = self.pf[9:12]
        pf_fr_b = mtimes(rotation.T, (pf_fr_w - self.p))
        pf_fl_b = mtimes(rotation.T, (pf_fl_w - self.p))
        pf_rr_b = mtimes(rotation.T, (pf_rr_w - self.p))
        pf_rl_b = mtimes(rotation.T, (pf_rl_w - self.p))
        self.y = vertcat(v_body, pf_fr_b, pf_fl_b, pf_rr_b, pf_rl_b)

        dp = self.v
        dv = -self.g * self.ez + mtimes(rotation, self.a)
        dq = 0.5 * self.quat_multiply_elementwise(
            self.q, vertcat(self.omega, DM(0.0))
        )
        dpf = vertcat(
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
            DM(0.0),
        )

        xdot = vertcat(dp, dv, dq, dpf)
        self.models_fn = Function(
            "models", [self.x, self.u], [xdot], ["x0", "u0"], ["xdot"]
        )

        k1 = self.noisy_rate(self.xa, self.u, self.w)
        k2 = self.noisy_rate(self.xa + self.dt / 2 * k1, self.u, self.w)
        k3 = self.noisy_rate(self.xa + self.dt / 2 * k2, self.u, self.w)
        k4 = self.noisy_rate(self.xa + self.dt * k3, self.u, self.w)
        self.models_mhe = (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def step(self, x, u, dt):
        k1 = self.models_fn(x0=x, u0=u)["xdot"].full()
        k2 = self.models_fn(x0=x + dt / 2 * k1, u0=u)["xdot"].full()
        k3 = self.models_fn(x0=x + dt / 2 * k2, u0=u)["xdot"].full()
        k4 = self.models_fn(x0=x + dt * k3, u0=u)["xdot"].full()
        xdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x_new = x + dt * xdot

        p_new = np.array([[x_new[0, 0], x_new[1, 0], x_new[2, 0]]]).T
        v_new = np.array([[x_new[3, 0], x_new[4, 0], x_new[5, 0]]]).T
        q_new = np.array([[x_new[6, 0], x_new[7, 0], x_new[8, 0], x_new[9, 0]]]).T
        rotation_new = self.quaternion_to_rotation_matrix(q_new)
        pf_new = np.array([x_new[10:22, 0]]).T

        gamma = np.arctan(rotation_new[2, 1] / rotation_new[1, 1])
        theta = np.arctan(rotation_new[0, 2] / rotation_new[0, 0])
        psi = np.arcsin(-rotation_new[0, 1])
        euler_new = np.array([[gamma, theta, psi]]).T
        return {
            "p_new": p_new,
            "v_new": v_new,
            "q_new": q_new,
            "pf_new": pf_new,
            "Euler": euler_new,
        }

    def skew_sym(self, v):
        return vertcat(
            horzcat(0, -v[2, 0], v[1, 0]),
            horzcat(v[2, 0], 0, -v[0, 0]),
            horzcat(-v[1, 0], v[0, 0], 0),
        )

    def omega_matrix(self, w):
        wx, wy, wz = w[0], w[1], w[2]
        return vertcat(
            horzcat(0, -wx, -wy, -wz),
            horzcat(wx, 0, wz, -wy),
            horzcat(wy, -wz, 0, wx),
            horzcat(wz, wy, -wx, 0),
        )

    def quat_multiply_elementwise(self, q1, q2):
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]

        x = w1 * x2 + w2 * x1 + (y1 * z2 - z1 * y2)
        y = w1 * y2 + w2 * y1 + (z1 * x2 - x1 * z2)
        z = w1 * z2 + w2 * z1 + (x1 * y2 - y1 * x2)
        w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2)
        return vertcat(x, y, z, w)

    def quaternion_to_rotation_matrix(self, q):
        q = q / sqrt(dot(q, q) + 1e-16)
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        return vertcat(
            horzcat(
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ),
            horzcat(
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx),
            ),
            horzcat(
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2),
            ),
        )

    def quaternion_discrete_update(self, q, omega, dt):
        qx, qy, qz, qw = q
        wx, wy, wz = omega
        dq = np.array(
            [
                qx + 0.5 * dt * (wx * qw - wy * qz + wz * qy),
                qy + 0.5 * dt * (wx * qz + wy * qw - wz * qx),
                qz + 0.5 * dt * (-wx * qy + wy * qx + wz * qw),
                qw - 0.5 * dt * (wx * qx + wy * qy + wz * qz),
            ]
        )
        return dq / np.linalg.norm(dq)
