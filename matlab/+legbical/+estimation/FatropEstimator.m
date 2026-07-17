classdef FatropEstimator < handle
    %FATROPESTIMATOR Lower OCP solve and KKT implicit differentiation.

    properties (SetAccess = private)
        graph
        lastResult = struct()
    end

    properties (Access = private)
        solver
        options
        kkt
        lastTheta = []
        lastPrimal = []
    end

    methods
        function obj = FatropEstimator(graph, options)
            obj.graph = graph;
            obj.options = options;
            obj.kkt = legbical.estimation.KktSystem(graph);
            solverOptions = struct('structure_detection', 'manual', ...
                'N', graph.transitionCount, 'nx', graph.nx, ...
                'nu', graph.nu, 'ng', graph.ng, 'expand', false, ...
                'print_time', false, 'fatrop', struct('print_level', 0, ...
                'tol', options.FatropTolerance, ...
                'max_iter', options.FatropMaxIterations));
            obj.solver = casadi.nlpsol( ...
                'fast_fie', 'fatrop', graph.nlp(), solverOptions);
        end

        function result = solve(obj, theta)
            theta = theta(:);
            if isequal(theta, obj.lastTheta)
                result = obj.lastResult;
                return;
            end
            if isempty(obj.lastPrimal)
                initial = obj.graph.initialGuess(theta);
            else
                initial = obj.lastPrimal;
            end
            solution = obj.solver('x0', initial, 'p', theta, ...
                'lbg', zeros(obj.graph.constraintSize, 1), ...
                'ubg', zeros(obj.graph.constraintSize, 1));
            primal = full(solution.x);
            costate = full(solution.lam_g);
            point = [primal; costate];
            state = obj.graph.stateFromPrimal(primal);
            result = struct('theta', theta, 'point', point, ...
                'primal', primal, 'costate', costate, 'state', state, ...
                'objective', full(solution.f), ...
                'kktInfNorm', norm(obj.graph.kktResidual(point, theta), inf), ...
                'constraintInfNorm', norm( ...
                obj.graph.constraints(primal, theta), inf));
            stats = obj.solver.stats();
            if ~stats.success || result.kktInfNorm > obj.options.KktTolerance ...
                    || result.constraintInfNorm > obj.options.ConstraintTolerance
                error('legbical:FatropFailure', ...
                    'Fatrop failed: KKT %.3g, constraint %.3g.', ...
                    result.kktInfNorm, result.constraintInfNorm);
            end
            obj.lastTheta = theta;
            obj.lastPrimal = primal;
            obj.lastResult = result;
        end

        function gradient = pullback(obj, result, lossStateGradient)
            obj.kkt.prepare(result.point, result.theta);
            selector = obj.graph.stateSelector();
            rhs = [selector' * lossStateGradient(:); ...
                zeros(obj.graph.constraintSize, 1)];
            adjoint = obj.kkt.solveTranspose(rhs);
            gradient = -full(obj.graph.kktTheta( ...
                result.point, result.theta)' * adjoint);
        end
    end
end
