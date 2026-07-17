classdef FIEGraph < handle
    %FIEGRAPH Stage-structured full-information estimator used by Fatrop.

    properties (Constant)
        StateSize = 8
        NoiseSize = 8
        ThetaSize = 24
    end

    properties (SetAccess = private)
        sampleCount
        transitionCount
        primalSize
        constraintSize
        statePrimalIndices
        nx
        nu
        ng
    end

    properties (Access = private)
        dt
        q
        dq
        ddq
        contact
        arrivalState
        kinematics
        options
        thetaSymbol
        primalSymbol
        objectiveExpression
        constraintExpression
        objectiveFunction
        constraintFunction
        kktResidualFunction
        kktMatrixFunction
        kktThetaFunction
    end

    methods
        function obj = FIEGraph(data, options)
            obj.dt = data.dt;
            obj.q = data.q;
            obj.dq = data.dq;
            obj.ddq = data.ddq;
            obj.contact = data.contact(:).';
            obj.kinematics = data.kinematics;
            obj.options = options;
            obj.sampleCount = size(obj.q, 2);
            obj.transitionCount = obj.sampleCount - 1;
            obj.arrivalState = zeros(obj.StateSize, 1);
            obj.arrivalState(1:2) = obj.q(1:2, 1);
            obj.arrivalState(3:4) = obj.dq(1:2, 1);
            obj.build();
        end

        function nlp = nlp(obj)
            nlp = struct('x', obj.primalSymbol, 'p', obj.thetaSymbol, ...
                'f', obj.objectiveExpression, 'g', obj.constraintExpression);
        end

        function value = constraints(obj, primal, theta)
            value = full(obj.constraintFunction(primal, theta(:)));
        end

        function value = kktResidual(obj, point, theta)
            value = full(obj.kktResidualFunction(point, theta(:)));
        end

        function value = kktMatrix(obj, point, theta)
            value = sparse(obj.kktMatrixFunction(point, theta(:)));
        end

        function value = kktTheta(obj, point, theta)
            value = sparse(obj.kktThetaFunction(point, theta(:)));
        end

        function state = stateFromPrimal(obj, primal)
            state = reshape(primal(obj.statePrimalIndices), ...
                obj.StateSize, obj.sampleCount);
        end

        function selector = stateSelector(obj)
            count = obj.StateSize * obj.sampleCount;
            selector = sparse(1:count, obj.statePrimalIndices, 1, ...
                count, obj.primalSize);
        end

        function guess = initialGuess(obj, theta)
            guess = zeros(obj.primalSize, 1);
            cursor = 0;
            for k = 1:obj.sampleCount
                state = [obj.q(1:2, k); obj.dq(1:2, k); ...
                    obj.q(1:2, k) + obj.position(k, true, theta); ...
                    obj.q(1:2, k) + obj.position(k, false, theta)];
                guess(cursor + (1:obj.StateSize)) = state;
                cursor = cursor + obj.StateSize;
                if k <= obj.transitionCount
                    cursor = cursor + obj.NoiseSize;
                end
            end
        end
    end

    methods (Access = private)
        function build(obj)
            import casadi.*
            K = obj.sampleCount;
            theta = MX.sym('theta', obj.ThetaSize, 1);
            states = cell(K, 1);
            noises = cell(K - 1, 1);
            blocks = cell(2 * K - 1, 1);
            indices = zeros(obj.StateSize * K, 1);
            blockCursor = 0;
            primalCursor = 0;
            stateCursor = 0;
            for k = 1:K
                states{k} = MX.sym(sprintf('x_%d', k - 1), ...
                    obj.StateSize, 1);
                blockCursor = blockCursor + 1;
                blocks{blockCursor} = states{k};
                indices(stateCursor + (1:obj.StateSize)) = ...
                    primalCursor + (1:obj.StateSize);
                stateCursor = stateCursor + obj.StateSize;
                primalCursor = primalCursor + obj.StateSize;
                if k < K
                    noises{k} = MX.sym(sprintf('w_%d', k - 1), ...
                        obj.NoiseSize, 1);
                    blockCursor = blockCursor + 1;
                    blocks{blockCursor} = noises{k};
                    primalCursor = primalCursor + obj.NoiseSize;
                end
            end

            primal = vertcat(blocks{:});
            objective = obj.arrivalCost(states{1}, theta);
            constraints = cell(K - 1, 1);
            for k = 1:K
                objective = objective + obj.measurementCost( ...
                    states{k}, theta, k);
                if k < K
                    objective = objective + obj.processCost( ...
                        noises{k}, theta, k);
                    constraints{k} = states{k + 1} ...
                        - obj.dynamics(states{k}, noises{k}, k);
                end
            end
            constraint = vertcat(constraints{:});
            lambda = MX.sym('lambda', size(constraint, 1), 1);
            point = [primal; lambda];
            kkt = [gradient(objective + lambda' * constraint, primal); ...
                constraint];

            obj.thetaSymbol = theta;
            obj.primalSymbol = primal;
            obj.objectiveExpression = objective;
            obj.constraintExpression = constraint;
            obj.primalSize = size(primal, 1);
            obj.constraintSize = size(constraint, 1);
            obj.statePrimalIndices = indices;
            obj.nx = obj.StateSize * ones(1, K);
            obj.nu = [obj.NoiseSize * ones(1, K - 1), 0];
            obj.ng = zeros(1, K);
            obj.objectiveFunction = Function('fie_objective', ...
                {primal, theta}, {objective});
            obj.constraintFunction = Function('fie_constraints', ...
                {primal, theta}, {constraint});
            obj.kktResidualFunction = Function('fie_kkt', ...
                {point, theta}, {kkt});
            obj.kktMatrixFunction = Function('fie_kkt_matrix', ...
                {point, theta}, {jacobian(kkt, point)});
            obj.kktThetaFunction = Function('fie_kkt_theta', ...
                {point, theta}, {jacobian(kkt, theta)});
        end

        function value = arrivalCost(obj, state, theta)
            expected = [obj.arrivalState(1:4); ...
                obj.arrivalState(1:2) + obj.position(1, true, theta); ...
                obj.arrivalState(1:2) + obj.position(1, false, theta)];
            residual = state - expected;
            value = obj.diagonalCost(residual(1:2), theta(18:19)) ...
                + obj.diagonalCost(residual(3:4), theta(20:21)) ...
                + obj.diagonalCost(residual(5:6), theta(22:23)) ...
                + obj.diagonalCost(residual(7:8), theta(22:23));
        end

        function value = processCost(obj, noise, theta, k)
            [leftStd, rightStd] = obj.footProcessStd(theta, obj.contact(k));
            value = obj.quadraticCost(noise(1:2), obj.covariance(theta(1:3))) ...
                + obj.quadraticCost(noise(3:4), obj.covariance(theta(4:6))) ...
                + obj.diagonalCost(noise(5:6), leftStd) ...
                + obj.diagonalCost(noise(7:8), rightStd);
        end

        function value = measurementCost(obj, state, theta, k)
            pLeft = obj.position(k, true, theta);
            pRight = obj.position(k, false, theta);
            JLeft = obj.jacobian(k, true, theta);
            JRight = obj.jacobian(k, false, theta);
            JLeftJoint = JLeft(:, 4:5);
            JRightJoint = JRight(:, 2:3);
            positionCovariance = obj.covariance(theta(11:13));
            velocityCovariance = obj.covariance(theta(14:16));
            leftPositionCovariance = JLeftJoint * positionCovariance ...
                * JLeftJoint';
            rightPositionCovariance = JRightJoint * positionCovariance ...
                * JRightJoint';
            value = obj.quadraticCost( ...
                state(5:6) - state(1:2) - pLeft, leftPositionCovariance) ...
                + obj.quadraticCost( ...
                state(7:8) - state(1:2) - pRight, rightPositionCovariance);

            generalizedVelocity = obj.dq(3:7, k);
            leftResidual = -state(3:4) - JLeft * generalizedVelocity;
            rightResidual = -state(3:4) - JRight * generalizedVelocity;
            leftCovariance = obj.footVelocityCovariance(theta, pLeft, ...
                JLeftJoint, positionCovariance, velocityCovariance, ...
                obj.dq(3, k));
            rightCovariance = obj.footVelocityCovariance(theta, pRight, ...
                JRightJoint, positionCovariance, velocityCovariance, ...
                obj.dq(3, k));
            if obj.contact(k) == 1
                leftCovariance = diag(theta(9:10).^2);
            elseif obj.contact(k) == -1
                rightCovariance = diag(theta(9:10).^2);
            end
            value = value + obj.quadraticCost(leftResidual, leftCovariance) ...
                + obj.quadraticCost(rightResidual, rightCovariance);
        end

        function next = dynamics(obj, state, noise, k)
            acceleration = obj.ddq(1:2, k);
            position = state(1:2) + obj.dt * state(3:4) ...
                + 0.5 * obj.dt^2 * acceleration + obj.dt * noise(1:2) ...
                + 0.5 * obj.dt^2 * noise(3:4);
            velocity = state(3:4) + obj.dt * acceleration ...
                + obj.dt * noise(3:4);
            left = state(5:6) + obj.dt * noise(5:6);
            right = state(7:8) + obj.dt * noise(7:8);
            next = [position; velocity; left; right];
        end

        function value = footVelocityCovariance(~, theta, position, ...
                jointJacobian, positionCovariance, velocityCovariance, omega)
            skew = omega * [0, -1; 1, 0];
            G = [skew * jointJacobian, jointJacobian, ...
                [0, 1; -1, 0] * position];
            zero22 = casadi.MX.zeros(2, 2);
            zero21 = casadi.MX.zeros(2, 1);
            zero12 = casadi.MX.zeros(1, 2);
            inputCovariance = [positionCovariance, zero22, zero21; ...
                zero22, velocityCovariance, zero21; ...
                zero12, zero12, theta(17)^2];
            value = G * inputCovariance * G';
        end

        function [left, right] = footProcessStd(~, theta, contact)
            if contact == 1
                left = theta(9:10); right = theta(7:8);
            elseif contact == -1
                left = theta(7:8); right = theta(9:10);
            else
                left = theta(7:8); right = theta(7:8);
            end
        end

        function value = position(obj, k, left, theta)
            if left
                value = obj.kinematics.leftPosition0(:, k) ...
                    + obj.kinematics.leftPositionSlope(:, k) * theta(24);
            else
                value = obj.kinematics.rightPosition0(:, k) ...
                    + obj.kinematics.rightPositionSlope(:, k) * theta(24);
            end
        end

        function value = jacobian(obj, k, left, theta)
            if left
                value = obj.kinematics.leftJacobian0(:, :, k) ...
                    + obj.kinematics.leftJacobianSlope(:, :, k) * theta(24);
            else
                value = obj.kinematics.rightJacobian0(:, :, k) ...
                    + obj.kinematics.rightJacobianSlope(:, :, k) * theta(24);
            end
        end

        function value = quadraticCost(obj, residual, covariance)
            covariance = 0.5 * (covariance + covariance') ...
                + obj.options.CovarianceJitter * eye(2);
            value = 0.5 * residual' * (covariance \ residual);
        end

        function value = diagonalCost(obj, residual, standardDeviation)
            variance = standardDeviation.^2 + obj.options.CovarianceJitter;
            value = 0.5 * sum(residual.^2 ./ variance);
        end
    end

    methods (Static, Access = private)
        function value = covariance(entries)
            value = [entries(1), entries(2); entries(2), entries(3)];
        end
    end
end
