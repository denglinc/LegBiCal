classdef SqpBfgsOptimizer < handle
    %SQPBFGSOPTIMIZER Feasible-coordinate SQP with BFGS curvature updates.

    properties (SetAccess = private)
        problem
        options
    end

    methods
        function obj = SqpBfgsOptimizer(problem, options)
            obj.problem = problem;
            obj.options = options;
        end

        function result = run(obj, theta0)
            map = legbical.calibration.CovarianceParameterization(obj.options);
            y0 = map.encode(theta0);
            cachedY = [];
            cached = struct();
            lossHistory = [];
            thetaHistory = [];
            settings = optimoptions('fmincon', 'Algorithm', 'sqp', ...
                'SpecifyObjectiveGradient', true, ...
                'SpecifyConstraintGradient', true, 'Display', 'off', ...
                'MaxIterations', obj.options.MaxIterations, ...
                'MaxFunctionEvaluations', 2000, ...
                'OptimalityTolerance', 1e-8, ...
                'ConstraintTolerance', 1e-10, ...
                'OutputFcn', @record);
            [yFinal, ~, exitflag, output] = fmincon(@objective, y0, ...
                [], [], [], [], zeros(size(y0)), ones(size(y0)), ...
                @constraints, settings);
            final = valueAt(yFinal, true);
            result = summary('sqp-bfgs', final, lossHistory, ...
                thetaHistory, exitflag, output.iterations, ...
                obj.problem.evaluations);

            function [loss, gradient] = objective(y)
                value = valueAt(y, true);
                loss = value.loss;
                [~, jacobian] = map.decode(y);
                gradient = jacobian' * value.gradient;
            end

            function value = valueAt(y, needGradient)
                if isequal(y, cachedY)
                    value = cached;
                    if needGradient && isempty(value.gradient)
                        value = obj.problem.differentiate(value);
                    end
                else
                    value = obj.problem.evaluate(map.decode(y), needGradient);
                    cachedY = y;
                end
                cached = value;
            end

            function [c, ceq, dc, dceq] = constraints(y)
                [c, ceq, dc, dceq] = map.constraints(y);
            end

            function stop = record(y, state, event)
                stop = false;
                if strcmp(event, 'init') || strcmp(event, 'iter')
                    thetaHistory(end + 1, :) = map.decode(y).';
                    lossHistory(end + 1, 1) = state.fval;
                end
            end
        end
    end
end

function result = summary(method, final, lossHistory, thetaHistory, ...
        status, iterations, evaluations)
result = struct('method', method, 'theta', final.theta, ...
    'loss', final.loss, 'gradient', final.gradient, ...
    'state', final.estimation.state, ...
    'lossHistory', lossHistory, 'thetaHistory', thetaHistory, ...
    'status', status, 'iterations', iterations, ...
    'evaluations', evaluations, 'lower', final.estimation);
end
