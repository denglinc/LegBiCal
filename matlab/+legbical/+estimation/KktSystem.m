classdef KktSystem < handle
    %KKTSYSTEM Sparse factorization used by the implicit-gradient adjoint.

    properties (SetAccess = private)
        graph
        point = []
        theta = []
        factorization
    end

    methods
        function obj = KktSystem(graph)
            obj.graph = graph;
        end

        function prepare(obj, point, theta)
            if isequal(point, obj.point) && isequal(theta, obj.theta)
                return;
            end
            matrix = obj.graph.kktMatrix(point, theta);
            matrix = 0.5 * (matrix + matrix');
            obj.factorization = decomposition(matrix, 'ldl');
            obj.point = point;
            obj.theta = theta;
        end

        function value = solveTranspose(obj, rhs)
            state = warning('query', 'MATLAB:nearlySingularMatrix');
            warning('off', 'MATLAB:nearlySingularMatrix');
            cleanup = onCleanup(@() warning(state));
            value = obj.factorization \ rhs;
            clear cleanup;
        end
    end
end
