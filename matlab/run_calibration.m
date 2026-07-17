function result = run_calibration(args)
%RUN_CALIBRATION Calibrate the fast FIE with a selected upper-level update.

arguments
    args.Method (1,1) string {mustBeMember(args.Method, ...
        ["sqp", "frank-wolfe", "adam"])} = "sqp"
    args.Horizon (1,1) string {mustBeMember(args.Horizon, ...
        ["demo", "full"])} = "demo"
    args.Iterations (1,1) double {mustBeInteger,mustBePositive} = 26
    args.DataFile (1,1) string = ""
end

setup();
root = fileparts(mfilename('fullpath'));
if args.DataFile == ""
    args.DataFile = fullfile(root, 'data', 'stride_demo.mat');
end
bundle = load(args.DataFile, 'fast');
if args.Horizon == "demo", count = 304; else, count = numel(bundle.fast.t); end
data = prefix(bundle.fast, count);
settings = legbical.config.options();
settings.MaxIterations = args.Iterations;

graph = legbical.estimation.FIEGraph(data, settings);
estimator = legbical.estimation.FatropEstimator(graph, settings);
problem = legbical.calibration.CalibrationProblem(estimator, ...
    data.groundTruth, settings.StateLossWeights);
switch args.Method
    case "sqp"
        optimizer = legbical.calibration.SqpBfgsOptimizer(problem, settings);
    case "frank-wolfe"
        optimizer = legbical.calibration.FrankWolfeOptimizer(problem, settings);
    case "adam"
        optimizer = legbical.calibration.ProjectedAdamOptimizer(problem, settings);
end
result = optimizer.run(settings.Theta0);
end

function data = prefix(source, count)
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
