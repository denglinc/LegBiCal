classdef ProjectedAdamOptimizer < handle
    %PROJECTEDADAMOPTIMIZER Adam in projected feasible coordinates.

    properties (SetAccess = private)
        problem
        options
    end

    methods
        function obj = ProjectedAdamOptimizer(problem, options)
            obj.problem = problem;
            obj.options = options;
        end

        function result = run(obj, theta0)
            map = legbical.calibration.CovarianceParameterization(obj.options);
            y = map.project(map.encode(theta0));
            firstMoment = zeros(size(y));
            secondMoment = zeros(size(y));
            lossHistory = nan(obj.options.MaxIterations, 1);
            thetaHistory = nan(obj.options.MaxIterations + 1, numel(theta0));
            thetaHistory(1, :) = theta0(:).';
            for k = 1:obj.options.MaxIterations
                [theta, jacobian] = map.decode(y);
                value = obj.problem.evaluate(theta, true);
                gradient = jacobian' * value.gradient;
                firstMoment = obj.options.AdamBeta1 * firstMoment ...
                    + (1 - obj.options.AdamBeta1) * gradient;
                secondMoment = obj.options.AdamBeta2 * secondMoment ...
                    + (1 - obj.options.AdamBeta2) * gradient.^2;
                correctedFirst = firstMoment / (1 - obj.options.AdamBeta1^k);
                correctedSecond = secondMoment / (1 - obj.options.AdamBeta2^k);
                y = map.project(y - obj.options.AdamLearningRate ...
                    * correctedFirst ./ (sqrt(correctedSecond) + 1e-8));
                lossHistory(k) = value.loss;
                thetaHistory(k + 1, :) = map.decode(y).';
            end
            final = obj.problem.evaluate(map.decode(y), true);
            result = struct('method', 'projected-adam', ...
                'theta', final.theta, 'loss', final.loss, ...
                'gradient', final.gradient, 'state', final.estimation.state, ...
                'lossHistory', lossHistory, 'thetaHistory', thetaHistory, ...
                'status', 1, 'iterations', obj.options.MaxIterations, ...
                'evaluations', obj.problem.evaluations, ...
                'lower', final.estimation);
        end
    end
end
