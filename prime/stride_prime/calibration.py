"""Upper-level loss and the three first-order update strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .estimator import PrimeEstimator, PrimeSolution
from .sensitivity import GaussNewtonKKTAdjoint


@dataclass(frozen=True)
class Evaluation:
    loss: float
    gradient: np.ndarray
    solution: PrimeSolution


class CalibrationProblem:
    def __init__(self, estimator: PrimeEstimator) -> None:
        self.estimator = estimator
        self.adjoint = GaussNewtonKKTAdjoint()
        self.evaluations = 0

    def evaluate(self, theta: np.ndarray) -> Evaluation:
        solution = self.estimator.solve(theta)
        loss, state_gradient = self.adjoint.upper_loss_and_state_gradient(solution)
        gradient = self.adjoint.differentiate(solution, state_gradient).gradient
        self.evaluations += 1
        return Evaluation(loss, gradient, solution)

    def run(self, method: str, theta0: np.ndarray, iterations: int) -> dict:
        if method == "sqp":
            return self._sqp(theta0, iterations)
        if method == "frank-wolfe":
            return self._frank_wolfe(theta0, iterations)
        if method == "adam":
            return self._adam(theta0, iterations)
        raise ValueError(f"unknown method: {method}")

    def _sqp(self, theta0: np.ndarray, iterations: int) -> dict:
        cache: dict[bytes, Evaluation] = {}
        losses: list[float] = []

        def value(theta: np.ndarray) -> Evaluation:
            key = np.asarray(theta, dtype=np.float64).tobytes()
            if key not in cache:
                cache[key] = self.evaluate(theta)
            return cache[key]

        def callback(theta: np.ndarray) -> None:
            losses.append(value(theta).loss)

        result = minimize(
            lambda theta: value(theta).loss, theta0,
            jac=lambda theta: value(theta).gradient,
            method="SLSQP", bounds=[(-2.5, 2.5)] * 4 + [(-0.05, 0.05)],
            options={"maxiter": iterations, "ftol": 1e-9}, callback=callback,
        )
        final = value(result.x)
        return self._summary(
            "sqp-bfgs", final, losses, int(result.nit), bool(result.success),
        )

    def _adam(self, theta0: np.ndarray, iterations: int) -> dict:
        theta = theta0.copy()
        first = np.zeros_like(theta)
        second = np.zeros_like(theta)
        losses: list[float] = []
        for k in range(1, iterations + 1):
            current = self.evaluate(theta)
            gradient = current.gradient.copy()
            norm = np.linalg.norm(gradient)
            if norm > 0.1:
                gradient *= 0.1 / norm
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient**2
            theta -= 0.08 * (first / (1 - 0.9**k)) / (
                np.sqrt(second / (1 - 0.999**k)) + 1e-8
            )
            theta[:4] = np.clip(theta[:4] * (1 - 8e-5), -2.5, 2.5)
            theta[4] = np.clip(theta[4], -0.05, 0.05)
            losses.append(current.loss)
        final = self.evaluate(theta)
        return self._summary(
            "projected-adam", final, losses, iterations, True,
        )

    def _frank_wolfe(self, theta0: np.ndarray, iterations: int) -> dict:
        theta = theta0.copy()
        lower = np.exp(np.array([-2.5] * 4 + [-0.05]))
        upper = np.exp(np.array([2.5] * 4 + [0.05]))
        current = self.evaluate(theta)
        losses: list[float] = []
        for _ in range(iterations):
            scales = np.exp(theta)
            gradient = current.gradient / scales
            vertex = np.where(gradient >= 0, lower, upper)
            direction = vertex - scales
            slope = float(gradient @ direction)
            if slope >= -1e-12 * max(1.0, np.linalg.norm(gradient)):
                break
            alpha = 1.0
            for _ in range(12):
                candidate_theta = np.log(scales + alpha * direction)
                candidate = self.evaluate(candidate_theta)
                if candidate.loss <= current.loss + 1e-4 * alpha * slope:
                    theta, current = candidate_theta, candidate
                    break
                alpha *= 0.5
            else:
                break
            losses.append(current.loss)
        return self._summary(
            "frank-wolfe", current, losses, len(losses), True,
        )

    def _summary(
        self, method: str, value: Evaluation, losses: list[float],
        iterations: int, success: bool,
    ) -> dict:
        return {
            "method": method,
            "theta": value.solution.theta,
            "loss": value.loss,
            "gradient": value.gradient,
            "state": value.solution.state,
            "loss_history": np.asarray(losses),
            "iterations": iterations,
            "evaluations": self.evaluations,
            "success": success,
        }
