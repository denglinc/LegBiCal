classdef CalibrationProblem < handle
    %CALIBRATIONPROBLEM Supervised trajectory loss and first-order oracle.

    properties (SetAccess = private)
        estimator
        groundTruth
        weights
        scale
        evaluations = 0
    end

    methods
        function obj = CalibrationProblem(estimator, groundTruth, weights)
            obj.estimator = estimator;
            obj.groundTruth = groundTruth;
            obj.weights = weights(:);
            obj.scale = 1 / size(groundTruth, 2);
        end

        function value = evaluate(obj, theta, gradientRequired)
            if nargin < 3, gradientRequired = true; end
            estimate = obj.estimator.solve(theta);
            [loss, derivative] = obj.lossDerivative(estimate.state);
            value = struct('theta', theta(:), 'loss', loss, ...
                'gradient', [], 'estimation', estimate);
            if gradientRequired
                value.gradient = obj.estimator.pullback(estimate, derivative);
            end
            obj.evaluations = obj.evaluations + 1;
        end

        function value = differentiate(obj, value)
            [~, derivative] = obj.lossDerivative(value.estimation.state);
            value.gradient = obj.estimator.pullback( ...
                value.estimation, derivative);
        end

        function [loss, derivative] = lossDerivative(obj, state)
            residual = state - obj.groundTruth;
            derivative = obj.scale * obj.weights .* residual;
            loss = 0.5 * sum(residual .* derivative, 'all');
        end
    end
end
