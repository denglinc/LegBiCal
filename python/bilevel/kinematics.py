import numpy as np
import pinocchio as pin
from scipy.linalg import block_diag
from .utils import skew, leg_joint_cols_from_J, unflatten_jac_colmajor


def _parent_joint(frame) -> int:
    """Return the parent joint across Pinocchio 2 and 3/4 APIs."""

    return int(frame.parentJoint if hasattr(frame, "parentJoint") else frame.parent)

def build_zeroed(q_i: np.ndarray, v_i: np.ndarray):
    """
    Zero out the floating-base pose/velocity while keeping joint part.
    q_i: full nq
    v_i: full nv
    """
    q_zero = q_i.copy()
    v_zero = v_i.copy()
    # world position = 0
    q_zero[0:3] = [0, 0, 0]
    # base orientation = identity quat
    q_zero[3:7] = [0, 0, 0, 1]
    # base linear velocity = 0
    v_zero[0:3] = [0, 0, 0]
    # base angular velocity = 0
    v_zero[3:6] = [0, 0, 0]
    return q_zero, v_zero

def compute_pf_meas(model, data, q_zero, fids, offset_calf):
    """
    Forward kinematics of feet in world frame, then add per-leg calf offset (expressed in parent joint frame).
    Returns:
      pf_i: (12,) stacked feet positions in world
      J_pf_off: (12, 12) block-diag of parent-joint rotations: ∂pf/∂offset
    """
    pin.forwardKinematics(model, data, q_zero)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_zero)

    pf_i = np.zeros(12, dtype=float)
    blocks = []
    for j, fid in enumerate(fids):
        jid = _parent_joint(model.frames[fid])
        Rj_i = np.asarray(data.oMi[jid].rotation, dtype=float)
        p_f  = np.asarray(data.oMf[fid].translation, dtype=float)
        r_off = offset_calf[3*j:3*j+3]

        pf_i[3*j:3*j+3] = p_f + Rj_i @ r_off
        blocks.append(Rj_i)

    J_pf_off = block_diag(*blocks)
    return pf_i, J_pf_off

def compute_yv_kin(model, data, q_zero, v_zero, omega_body, fids, offset_calf):
    """
    Compute foot velocity measurement *in the style you used*:
    - base-part: -(J_lin * qdot + omega x p_f)
    - then subtract the part from foot offset
    Also returns Jacobian wrt offset.
    """
    pin.forwardKinematics(model, data, q_zero, v_zero)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_zero)

    v_foot = np.zeros(12, dtype=float)
    omega_body = np.asarray(omega_body, dtype=float).reshape(3,)
    blocks = []

    for j, fid in enumerate(fids):
        # base-part w/o offset
        J_lwa = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_lin = np.asarray(J_lwa[0:3, 6:18])
        p_f   = np.asarray(data.oMf[fid].translation, dtype=float)
        v_foot[3*j:3*j+3] = -((J_lin @ v_zero[6:18]) + np.cross(omega_body, p_f))

        # offset-part
        jid   = _parent_joint(model.frames[fid])
        Rj_i  = np.asarray(data.oMi[jid].rotation, dtype=float)
        r_off = offset_calf[3*j:3*j+3]
        pf_off = Rj_i @ r_off

        Jj_i = np.asarray(pin.getJointJacobian(model, data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
        omega_joint = (Jj_i[3:6, :] @ v_zero).reshape(3,)

        # v_off = -skew(pf_off) @ (omega_joint) + omega_body × pf_off
        v_off = (-pin.skew(pf_off) @ (Jj_i[3:6, :] @ v_zero)).reshape(3,) + np.cross(omega_body, pf_off)

        v_foot[3*j:3*j+3] += -v_off

        # ∂y/∂offset = -(Skew(omega_joint)+Skew(omega_body)) @ Rj_i
        J_block = -(np.asarray(pin.skew(omega_joint)) @ Rj_i + np.asarray(pin.skew(omega_body)) @ Rj_i)
        blocks.append(J_block)

    J_v_off = block_diag(*blocks)
    return v_foot, J_v_off

def compute_Gv_leg_blocks_body(model, data, q_zero, v_zero, omega, fids, eps=1e-7):
    """
    Numerical differentiation way to build velocity part of G (per leg, 3x9).
    Matches your original logic.
    """
    qB = q_zero.copy()
    qB[0:3] = 0.0
    qB[3:7] = np.array([0,0,0,1.0])
    nJ = model.nv - 6
    qJ_idx = np.arange(7, 7+nJ)
    qdJ = v_zero[6:6+nJ].copy()

    pin.forwardKinematics(model, data, qB)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, qB)

    G_list = []
    for fid in fids:
        J6 = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J  = np.asarray(J6[0:3, 6:6+nJ])
        f  = np.asarray(data.oMf[fid].translation)
        leg_cols = leg_joint_cols_from_J(J)
        if leg_cols.size != 3:
            raise RuntimeError(f"leg has {leg_cols.size} cols, expected 3.")

        DqJdot_full = np.zeros((3, nJ))
        for i in range(nJ):
            q_p = qB.copy(); q_p[qJ_idx[i]] += eps
            q_m = qB.copy(); q_m[qJ_idx[i]] -= eps

            pin.computeJointJacobians(model, data, q_p); pin.updateFramePlacements(model, data)
            Jp = np.asarray(pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6:6+nJ])

            pin.computeJointJacobians(model, data, q_m); pin.updateFramePlacements(model, data)
            Jm = np.asarray(pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3, 6:6+nJ])

            dJ_dqi = (Jp - Jm) / (2.0*eps)
            DqJdot_full += dJ_dqi * qdJ[i]

        dvdq_leg    = (DqJdot_full + skew(omega) @ J)[:, leg_cols]
        dvdqdot_leg = J[:, leg_cols]
        dvdw        = -skew(f)

        G_leg = np.hstack([dvdq_leg, dvdqdot_leg, dvdw])  # 3x9
        G_list.append(G_leg)

    return G_list

def compute_Gp_leg_blocks_body(model, data, q_zero, fids):
    """
    Position part of G: just foot position jacobian w.r.t. 3 leg joints.
    """
    qB = q_zero.copy()
    qB[0:3] = 0.0
    qB[3:7] = np.array([0,0,0,1.0])
    nJ = model.nv - 6
    pin.forwardKinematics(model, data, qB)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, qB)

    Gp_list = []
    for fid in fids:
        J6 = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J  = np.asarray(J6[0:3, 6:6+nJ])
        leg_cols = leg_joint_cols_from_J(J)
        if leg_cols.size != 3:
            raise RuntimeError(f"leg has {leg_cols.size} cols, expected 3.")
        J_leg = J[:, leg_cols]
        Gp_list.append(np.hstack([J_leg, np.zeros((3,3)), np.zeros((3,3))]))
    return Gp_list

def pack_G24x9(Gv_leg_list, Gp_leg_list):
    blocks = []
    for G in Gv_leg_list: blocks.append(G)
    for G in Gp_leg_list: blocks.append(G)
    return np.vstack(blocks)  # (24,9)

def dGk_dtip_from_codegen(f_yv, f_pf, model, data, q_zero, v_zero, omega, fids):
    """
    This is the de-duplicated version of your original function.
    It calls two CasADi generated externals:
      - f_yv: gives derivatives of foot velocity jacobians wrt tip offset
      - f_pf: gives derivatives of foot position jacobians wrt tip offset
    Output:
      (24*9, 12) i.e. (216, 12) flattened in FORTRAN order
    """
    nq, nv = model.nq, model.nv
    q_joint = np.asarray(q_zero[7:], dtype=float)
    v_joint = np.asarray(v_zero[6:], dtype=float)
    omega   = np.asarray(omega,       dtype=float)
    theta0  = np.zeros(12, dtype=float)

    out_v = f_yv(q=q_joint, v=v_joint, omega=omega, theta=theta0)
    dJq_flat  = np.array(out_v["dJy_q_dtheta"])
    dJv_flat  = np.array(out_v["dJy_v_dtheta"])
    dJw_flat  = np.array(out_v["dJy_omega_dtheta"])

    rows_v = 12
    cols_q = (nq - 7)
    cols_v = (nv - 6)
    cols_w = 3
    p      = 12

    dJ_q_tensor  = unflatten_jac_colmajor(dJq_flat, rows_v, cols_q)
    dJ_v_tensor  = unflatten_jac_colmajor(dJv_flat, rows_v, cols_v)
    dJ_w_tensor  = unflatten_jac_colmajor(dJw_flat, rows_v, cols_w)

    out_p    = f_pf(q=q_joint, theta=theta0)
    dJp_flat = np.array(out_p["dJy_dtheta"])
    dJ_p_tensor = unflatten_jac_colmajor(dJp_flat, rows_v, cols_q)

    # find 3 active cols per leg
    pin.forwardKinematics(model, data, q_zero)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_zero)

    nJ = nv - 6
    leg_cols_all = []
    for fid in fids:
        J6  = pin.getFrameJacobian(model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J   = np.asarray(J6[0:3, 6:6+nJ])
        cols = leg_joint_cols_from_J(J)
        if cols.size != 3:
            raise RuntimeError(f"leg has {cols.size} cols, expected 3.")
        leg_cols_all.append(cols)

    dG_all = np.zeros((24, 9, p))

    # velocity rows
    for L, cols in enumerate(leg_cols_all):
        r0 = 3*L
        dG_all[r0:r0+3, 0:3, :] = dJ_q_tensor[r0:r0+3, cols, :]
        dG_all[r0:r0+3, 3:6, :] = dJ_v_tensor[r0:r0+3, cols, :]
        dG_all[r0:r0+3, 6:9, :] = dJ_w_tensor[r0:r0+3, :, :]

    # position rows
    for L, cols in enumerate(leg_cols_all):
        r0 = 12 + 3*L
        dG_all[r0:r0+3, 0:3, :] = dJ_p_tensor[r0-12:r0-12+3, cols, :]

    return np.reshape(dG_all, (24*9, p), order='F')
