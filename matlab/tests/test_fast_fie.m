function test_fast_fie()
%TEST_FAST_FIE Numerical smoke and directional-gradient checks.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
setup();
bundle = load(fullfile(root, 'data', 'stride_demo.mat'), 'fast');
data = subset(bundle.fast, 80);
settings = legbical.config.options();
settings.MaxIterations = 1;
graph = legbical.estimation.FIEGraph(data, settings);
estimator = legbical.estimation.FatropEstimator(graph, settings);
problem = legbical.calibration.CalibrationProblem( ...
    estimator, data.groundTruth, settings.StateLossWeights);

theta = settings.Theta0;
value = problem.evaluate(theta, true);
step = 1e-4;
plus = theta; plus(24) = plus(24) + step;
minus = theta; minus(24) = minus(24) - step;
finiteDifference = (problem.evaluate(plus, false).loss ...
    - problem.evaluate(minus, false).loss) / (2 * step);
relativeError = abs(value.gradient(24) - finiteDifference) ...
    / max([1, abs(value.gradient(24)), abs(finiteDifference)]);
assert(relativeError < 1e-3);

methods = {@legbical.calibration.SqpBfgsOptimizer, ...
    @legbical.calibration.FrankWolfeOptimizer, ...
    @legbical.calibration.ProjectedAdamOptimizer};
for k = 1:numel(methods)
    optimizer = methods{k}(problem, settings);
    result = optimizer.run(theta);
    assert(isfinite(result.loss) && all(isfinite(result.theta)));
end
end

function data = subset(source, count)
data = struct('q', source.q(:, 1:count), ...
    'dq', source.dq(:, 1:count), 'ddq', source.ddq(:, 1:count), ...
    'contact', source.contact(1:count), 't', source.t(1:count), ...
    'dt', source.dt, 'groundTruth', source.groundTruth(:, 1:count));
names = fieldnames(source.kinematics);
for k = 1:numel(names)
    value = source.kinematics.(names{k});
    if ndims(value) == 2
        data.kinematics.(names{k}) = value(:, 1:count);
    else
        data.kinematics.(names{k}) = value(:, :, 1:count);
    end
end
end
