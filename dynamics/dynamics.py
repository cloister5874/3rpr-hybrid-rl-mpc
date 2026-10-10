from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import sympy as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


@dataclass
class RobotParams:
    rA: Tuple[float, float] = (0.0, 0.0)
    rB: Tuple[float, float] = (1.0, 0.0)
    rC: Tuple[float, float] = (1.0, 2.0)
    a: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    b: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    c12: float = 0.15
    c23: float = 0.15
    c31: float = 0.15
    ma: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mG: float = 1.0
    Ia: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    Ib: Tuple[float, float, float] = (0.01, 0.01, 0.01)
    IG: float = 0.01
    la: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    lb: Tuple[float, float, float] = (0.15, 0.15, 0.15)


def load_params_from_robot_properties() -> RobotParams:
    try:
        import robot_properties as rp  # type: ignore
        raw = rp.robot_parameters()
        inert = rp.inertial_and_geometrical_parameters(raw)

        def get_any(d: Dict, *names, default=None):
            for n in names:
                if n in d:
                    return d[n]
            return default

        rA = (float(get_any(raw, "rax", "rAx", "r_Ax", default=0.0)),
              float(get_any(raw, "ray", "rAy", "r_Ay", default=0.0)))
        rB = (float(get_any(raw, "rbx", "rBx", "r_Bx", default=1.0)),
              float(get_any(raw, "rby", "rBy", "r_By", default=0.0)))
        rC = (float(get_any(raw, "rcx", "rCx", "r_Cx", default=1.0)),
              float(get_any(raw, "rcy", "rCy", "r_Cy", default=2.0)))

        a = tuple(float(get_any(raw, f"a{i}", default=0.1)) for i in [1, 2, 3])
        b = tuple(float(get_any(raw, f"b{i}", default=0.3)) for i in [1, 2, 3])
        c12 = float(get_any(raw, "c12", default=0.15))
        c23 = float(get_any(raw, "c23", default=0.15))
        c31 = float(get_any(raw, "c31", default=0.15))

        ma = tuple(float(get_any(inert, f"ma{i}", default=1.0)) for i in [1, 2, 3])
        mb = tuple(float(get_any(inert, f"mb{i}", default=1.0)) for i in [1, 2, 3])
        mG = float(get_any(inert, "mG", "mg", default=1.0))
        Ia = tuple(float(get_any(inert, f"Ia{i}", default=0.01)) for i in [1, 2, 3])
        Ib = tuple(float(get_any(inert, f"Ib{i}", default=0.01)) for i in [1, 2, 3])
        IG = float(get_any(inert, "IG", "Ig", default=0.01))
        la = tuple(float(get_any(inert, f"la{i}", default=0.05)) for i in [1, 2, 3])
        lb = tuple(float(get_any(inert, f"lb{i}", default=0.15)) for i in [1, 2, 3])

        print("Loaded parameters from robot_properties.py")
        return RobotParams(rA=rA, rB=rB, rC=rC, a=a, b=b,
                           c12=c12, c23=c23, c31=c31,
                           ma=ma, mb=mb, mG=mG,
                           Ia=Ia, Ib=Ib, IG=IG, la=la, lb=lb)
    except Exception as e:
        print("Could not load robot_properties.py. Using fallback parameters.")
        print("Reason:", repr(e))
        return RobotParams()


def platform_local_points_from_sides(c12: float, c23: float, c31: float):
    u = (c31**2 + c12**2 - c23**2) / (2.0 * c12)
    v = math.sqrt(max(c31**2 - u**2, 0.0))
    P = np.array([[0.0, 0.0], [c12, 0.0], [u, v]], dtype=float)
    P -= np.mean(P, axis=0)
    return P[0], P[1], P[2]


def build_sympy_model(params: RobotParams):
    d1, d2, d3 = sp.symbols("d1 d2 d3")
    th1, th2, th3 = sp.symbols("theta1 theta2 theta3")
    ph12, ph23, ph31 = sp.symbols("phi12 phi23 phi31")
    q_syms = [d1, d2, d3, th1, th2, th3, ph12, ph23, ph31]

    d1d, d2d, d3d = sp.symbols("d1_dot d2_dot d3_dot")
    th1d, th2d, th3d = sp.symbols("theta1_dot theta2_dot theta3_dot")
    ph12d, ph23d, ph31d = sp.symbols("phi12_dot phi23_dot phi31_dot")
    qd_syms = [d1d, d2d, d3d, th1d, th2d, th3d, ph12d, ph23d, ph31d]

    def e(alpha):
        return sp.Matrix([sp.cos(alpha), sp.sin(alpha)])

    rA = sp.Matrix(params.rA)
    rB = sp.Matrix(params.rB)
    rC = sp.Matrix(params.rC)

    a1, a2, a3 = params.a
    b1, b2, b3 = params.b
    c12, c23, c31 = params.c12, params.c23, params.c31

    f1 = (a1 + d1 + b1) * e(th1) + c12 * e(th1 + ph12) - (a2 + d2 + b2) * e(th2) - (rB - rA)
    f2 = (a2 + d2 + b2) * e(th2) + c23 * e(th2 + ph23) - (a3 + d3 + b3) * e(th3) - (rC - rB)
    f3 = (a3 + d3 + b3) * e(th3) + c31 * e(th3 + ph31) - (a1 + d1 + b1) * e(th1) - (rA - rC)

    Phi = sp.Matrix.vstack(f1, f2, f3)
    A = Phi.jacobian(sp.Matrix(q_syms))

    q_vec = sp.Matrix(q_syms)
    qd_vec = sp.Matrix(qd_syms)

    def time_derivative(vec):
        return vec.jacobian(q_vec) * qd_vec

    r_base = [rA, rB, rC]
    d = [d1, d2, d3]
    th = [th1, th2, th3]
    thd = [th1d, th2d, th3d]

    T = 0
    for i in range(3):
        r_ai = r_base[i] + params.la[i] * e(th[i])
        v_ai = time_derivative(r_ai)
        T += sp.Rational(1, 2) * params.ma[i] * v_ai.dot(v_ai) + sp.Rational(1, 2) * params.Ia[i] * thd[i]**2

        r_bi = r_base[i] + (params.a[i] + d[i] + params.lb[i]) * e(th[i])
        v_bi = time_derivative(r_bi)
        T += sp.Rational(1, 2) * params.mb[i] * v_bi.dot(v_bi) + sp.Rational(1, 2) * params.Ib[i] * thd[i]**2

    P1 = rA + (a1 + d1 + b1) * e(th1)
    P2 = rB + (a2 + d2 + b2) * e(th2)
    P3 = rC + (a3 + d3 + b3) * e(th3)
    G = (P1 + P2 + P3) / 3
    vG = time_derivative(G)
    psi_dot = th1d + ph12d
    T += sp.Rational(1, 2) * params.mG * vG.dot(vG) + sp.Rational(1, 2) * params.IG * psi_dot**2
    L = T

    modules = [{"sin": jnp.sin, "cos": jnp.cos, "sqrt": jnp.sqrt, "atan2": jnp.arctan2}, "jax"]
    Phi_fun = sp.lambdify(q_syms, Phi, modules=modules)
    A_fun = sp.lambdify(q_syms, A, modules=modules)
    L_fun = sp.lambdify(q_syms + qd_syms, L, modules=modules)
    return Phi_fun, A_fun, L_fun


def make_reduced_dynamics(params: RobotParams, Phi_fun, A_fun, L_fun):
    p1, p2, p3 = platform_local_points_from_sides(params.c12, params.c23, params.c31)
    p_local = jnp.array(np.stack([p1, p2, p3]), dtype=jnp.float64)
    base = jnp.array([params.rA, params.rB, params.rC], dtype=jnp.float64)
    a = jnp.array(params.a, dtype=jnp.float64)
    b = jnp.array(params.b, dtype=jnp.float64)

    def rot(psi):
        c = jnp.cos(psi)
        s = jnp.sin(psi)
        return jnp.array([[c, -s], [s, c]], dtype=jnp.float64)

    def q_from_x(x):
        xc, yc, psi = x
        R = rot(psi)
        center = jnp.array([xc, yc], dtype=jnp.float64)
        P = center[None, :] + (R @ p_local.T).T
        V = P - base

        lengths = jnp.linalg.norm(V, axis=1)
        d = lengths - a - b
        theta = jnp.arctan2(V[:, 1], V[:, 0])

        side12 = P[1] - P[0]
        side23 = P[2] - P[1]
        side31 = P[0] - P[2]

        alpha12 = jnp.arctan2(side12[1], side12[0])
        alpha23 = jnp.arctan2(side23[1], side23[0])
        alpha31 = jnp.arctan2(side31[1], side31[0])

        phi12 = alpha12 - theta[0]
        phi23 = alpha23 - theta[1]
        phi31 = alpha31 - theta[2]
        return jnp.concatenate([d, theta, jnp.array([phi12, phi23, phi31], dtype=jnp.float64)])

    def L_full(q, qd):
        return jnp.squeeze(jnp.array(L_fun(*(tuple(q) + tuple(qd))), dtype=jnp.float64))

    def L_reduced(x, xd):
        q = q_from_x(x)
        DqDx = jax.jacfwd(q_from_x)(x)
        qd = DqDx @ xd
        return L_full(q, qd)

    def DqaDx_fun(x):
        return jax.jacfwd(q_from_x)(x)[:3, :]

    def reduced_matrices(x, xd):
        B = jax.hessian(lambda vv: L_reduced(x, vv))(xd)
        C_L = jax.grad(lambda xx: L_reduced(xx, xd))(x)

        def dL_dxdot(xx):
            return jax.grad(lambda vv: L_reduced(xx, vv))(xd)

        A_L = jax.jacfwd(dL_dxdot)(x)
        return B, A_L, C_L

    @jax.jit
    def state_derivative(s, tau):
        x = s[:3]
        xd = s[3:]
        tau = jnp.asarray(tau, dtype=jnp.float64)
        B, A_L, C_L = reduced_matrices(x, xd)
        DqaDx = DqaDx_fun(x)
        Qx = DqaDx.T @ tau
        xdd = jnp.linalg.solve(B, Qx + C_L - A_L @ xd)
        return jnp.concatenate([xd, xdd])

    def platform_force_to_tau(x, Qx):
        DqaDx = DqaDx_fun(x)
        return jnp.linalg.solve(DqaDx.T, Qx)

    def diagnostics(s):
        x = jnp.asarray(s[:3], dtype=jnp.float64)
        xd = jnp.asarray(s[3:], dtype=jnp.float64)
        q = q_from_x(x)
        DqDx = jax.jacfwd(q_from_x)(x)
        qd = DqDx @ xd

        Phi = jnp.ravel(jnp.array(Phi_fun(*tuple(q)), dtype=jnp.float64))
        A = jnp.array(A_fun(*tuple(q)), dtype=jnp.float64)
        Ap = A[:, 3:]
        DqaDx = DqDx[:3, :]
        J = jnp.linalg.inv(DqaDx)
        B, _, _ = reduced_matrices(x, xd)

        return {
            "q": np.array(q),
            "qd": np.array(qd),
            "phi_residual": float(jnp.linalg.norm(Phi)),
            "vel_residual": float(jnp.linalg.norm(A @ qd)),
            "cond_Ap": float(jnp.linalg.cond(Ap)),
            "cond_J": float(jnp.linalg.cond(J)),
            "cond_B": float(jnp.linalg.cond(B)),
        }

    return q_from_x, L_reduced, state_derivative, platform_force_to_tau, diagnostics

params = load_params_from_robot_properties()
Phi_fun, A_fun, L_fun = build_sympy_model(params)
_, _, state_derivative, _, _ = make_reduced_dynamics(params, Phi_fun, A_fun, L_fun)
s_test = jnp.array([0.45, 0.45, 0.10, 0.10, -0.05, 0.02], dtype=jnp.float64)
tau_test = jnp.array([0.5, -0.25, 0.15], dtype=jnp.float64)
sd = state_derivative(s_test, tau_test)

def step(state, forces, dt=0.01):
    """
    One simulation step of the 3-RPR robot.

    state  = [x, y, theta, x_dot, y_dot, theta_dot]
    forces = [F1, F2, F3]
    dt     = simulation time step
    """

    # Calculate state derivative
    state_dot = state_derivative(state, forces)

    # Euler integration
    next_state = state + dt * state_dot

    return next_state

def calculate_baseline_reward(next_state, desired_state, forces, lam=0.01):
    # State error
    state_error = next_state - desired_state

    # Squared state error
    state_cost = np.sum(state_error ** 2)

    # Squared force magnitude
    force_cost = np.sum(forces ** 2)

    # Baseline reward from the supplied paper
    reward = -state_cost - lam * force_cost

    return reward

