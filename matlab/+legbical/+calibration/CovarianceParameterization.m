classdef CovarianceParameterization
    %COVARIANCEPARAMETERIZATION Scaled Cholesky coordinates for theta.

    properties (SetAccess = private)
        lower
        upper
        blocks
        epsilon
        latentLower
        latentUpper
        latentScale
    end

    methods
        function obj = CovarianceParameterization(options)
            obj.lower = options.LowerBound(:);
            obj.upper = options.UpperBound(:);
            obj.blocks = [1, 2, 3; 4, 5, 6; 11, 12, 13; 14, 15, 16];
            obj.epsilon = options.PsdEpsilon;
            obj.latentLower = obj.lower;
            obj.latentUpper = obj.upper;
            for block = 1:size(obj.blocks, 1)
                index = obj.blocks(block, :);
                radius = sqrt(obj.upper(index(3)) - obj.epsilon);
                obj.latentLower(index) = [1e-8; -radius; 1e-8];
                obj.latentUpper(index) = [sqrt( ...
                    obj.upper(index(1)) - obj.epsilon); radius; radius];
            end
            obj.latentScale = obj.latentUpper - obj.latentLower;
        end

        function y = encode(obj, theta)
            z = theta(:);
            for block = 1:size(obj.blocks, 1)
                index = obj.blocks(block, :);
                l11 = sqrt(max(theta(index(1)) - obj.epsilon, eps));
                l21 = theta(index(2)) / l11;
                l22 = sqrt(max(theta(index(3)) - obj.epsilon - l21^2, eps));
                z(index) = [l11; l21; l22];
            end
            y = (z - obj.latentLower) ./ obj.latentScale;
        end

        function [theta, jacobian] = decode(obj, y)
            z = obj.latentLower + obj.latentScale .* y(:);
            theta = z;
            jacobian = diag(obj.latentScale);
            for block = 1:size(obj.blocks, 1)
                index = obj.blocks(block, :);
                l11 = z(index(1));
                l21 = z(index(2));
                l22 = z(index(3));
                theta(index) = [obj.epsilon + l11^2; ...
                    l11 * l21; obj.epsilon + l21^2 + l22^2];
                jacobian(index, :) = 0;
                jacobian(index(1), index(1)) = ...
                    2 * l11 * obj.latentScale(index(1));
                jacobian(index(2), index(1)) = ...
                    l21 * obj.latentScale(index(1));
                jacobian(index(2), index(2)) = ...
                    l11 * obj.latentScale(index(2));
                jacobian(index(3), index(2)) = ...
                    2 * l21 * obj.latentScale(index(2));
                jacobian(index(3), index(3)) = ...
                    2 * l22 * obj.latentScale(index(3));
            end
        end

        function y = project(obj, y)
            z = obj.latentLower + obj.latentScale .* min(1, max(0, y(:)));
            for block = 1:size(obj.blocks, 1)
                index = obj.blocks(block, :);
                l11 = min(obj.latentUpper(index(1)), ...
                    max(obj.latentLower(index(1)), z(index(1))));
                radius = sqrt(obj.upper(index(3)) - obj.epsilon);
                lowerL21 = max(-radius, obj.lower(index(2)) / l11);
                upperL21 = min(radius, obj.upper(index(2)) / l11);
                l21 = min(upperL21, max(lowerL21, z(index(2))));
                maxL22 = sqrt(max(radius^2 - l21^2, 1e-16));
                l22 = min(maxL22, max(1e-8, z(index(3))));
                z(index) = [l11; l21; l22];
            end
            y = (z - obj.latentLower) ./ obj.latentScale;
            y = min(1, max(0, y));
        end

        function [c, ceq, gradient, gradientEq] = constraints(obj, y)
            [theta, jacobian] = obj.decode(y);
            c = zeros(3 * size(obj.blocks, 1), 1);
            gradient = zeros(numel(theta), numel(c));
            for block = 1:size(obj.blocks, 1)
                index = obj.blocks(block, :);
                cursor = 3 * (block - 1);
                c(cursor + 1) = theta(index(3)) - obj.upper(index(3));
                c(cursor + 2) = theta(index(2)) - obj.upper(index(2));
                c(cursor + 3) = obj.lower(index(2)) - theta(index(2));
                gradient(:, cursor + 1) = jacobian(index(3), :).';
                gradient(:, cursor + 2) = jacobian(index(2), :).';
                gradient(:, cursor + 3) = -jacobian(index(2), :).';
            end
            ceq = [];
            gradientEq = [];
        end
    end
end
