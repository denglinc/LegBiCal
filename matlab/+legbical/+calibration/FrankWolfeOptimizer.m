classdef FrankWolfeOptimizer < handle
    %FRANKWOLFEOPTIMIZER Feasible updates from a semidefinite LMO.

    properties (SetAccess = private)
        problem
        options
        cones
    end

    methods
        function obj = FrankWolfeOptimizer(problem, options)
            obj.problem = problem;
            obj.options = options;
            obj.cones = obj.buildCones();
        end

        function result = run(obj, theta0)
            theta = theta0(:);
            lossHistory = nan(obj.options.MaxIterations, 1);
            thetaHistory = nan(obj.options.MaxIterations + 1, numel(theta));
            thetaHistory(1, :) = theta.';
            current = obj.problem.evaluate(theta, true);
            completed = 0;
            for k = 1:obj.options.MaxIterations
                vertex = obj.oracle(current.gradient, theta);
                direction = vertex - theta;
                slope = current.gradient' * direction;
                if slope >= -1e-12 * max(1, norm(current.gradient))
                    break;
                end
                [theta, current] = obj.lineSearch( ...
                    theta, direction, slope, current.loss);
                lossHistory(k) = current.loss;
                thetaHistory(k + 1, :) = theta.';
                completed = k;
            end
            lossHistory = lossHistory(1:completed);
            thetaHistory = thetaHistory(1:completed + 1, :);
            if isempty(current.gradient)
                current = obj.problem.differentiate(current);
            end
            result = struct('method', 'frank-wolfe', ...
                'theta', theta, 'loss', current.loss, ...
                'gradient', current.gradient, ...
                'state', current.estimation.state, ...
                'lossHistory', lossHistory, 'thetaHistory', thetaHistory, ...
                'status', 1, 'iterations', completed, ...
                'evaluations', obj.problem.evaluations, ...
                'lower', current.estimation);
        end
    end

    methods (Access = private)
        function vertex = oracle(obj, gradient, theta)
            lower = obj.options.LowerBound(:);
            upper = obj.options.UpperBound(:);
            radius = obj.options.TrustRegionFraction * (upper - lower);
            lower = max(lower, theta - radius);
            upper = min(upper, theta + radius);
            settings = optimoptions('coneprog', 'Display', 'off');
            [vertex, ~, flag] = coneprog(gradient, obj.cones, ...
                [], [], [], [], lower, upper, settings);
            if flag <= 0
                error('legbical:LinearOracle', ...
                    'The semidefinite linear oracle did not converge.');
            end
        end

        function [theta, value] = lineSearch(obj, theta, direction, slope, loss)
            alpha = 1;
            for k = 1:obj.options.ArmijoMaxSteps
                candidate = theta + alpha * direction;
                try
                    value = obj.problem.evaluate(candidate, false);
                catch failure
                    if strcmp(failure.identifier, 'legbical:FatropFailure')
                        alpha = obj.options.ArmijoBeta * alpha;
                        continue;
                    end
                    rethrow(failure);
                end
                if value.loss <= loss + obj.options.ArmijoRho * alpha * slope
                    theta = candidate;
                    value = obj.problem.differentiate(value);
                    return;
                end
                alpha = obj.options.ArmijoBeta * alpha;
            end
            error('legbical:LineSearch', 'Armijo line search did not converge.');
        end

        function cones = buildCones(obj)
            blocks = [1, 2, 3; 4, 5, 6; 11, 12, 13; 14, 15, 16];
            n = numel(obj.options.Theta0);
            for k = 1:size(blocks, 1)
                index = blocks(k, :);
                A = zeros(2, n);
                A(1, index(2)) = 2;
                A(2, index(1)) = 1;
                A(2, index(3)) = -1;
                c = zeros(n, 1);
                c(index([1, 3])) = 1;
                cone = secondordercone(A, zeros(2, 1), c, ...
                    -2 * obj.options.PsdEpsilon);
                if k == 1, cones = cone; else, cones(k) = cone; end
            end
        end
    end
end
