function value = options()
%OPTIONS Numerical settings shared by the estimator and upper optimizers.

value.Theta0 = [1e-4; 0; 1e-4; 1; 0; 1; 1e-3; 1e-3; 1e5; 1e5; ...
    1e-3; 0; 1e-3; 1e-4; 0; 1e-4; 1e-3; 1e-3; 1e-3; 1e-3; ...
    1e-3; 1e-3; 1e-3; 0];
value.LowerBound = 1e-5 * ones(24, 1);
value.UpperBound = ones(24, 1);
value.LowerBound([2, 5, 12, 15]) = -1;
value.UpperBound([2, 5, 12, 15]) = 1;
value.UpperBound(4:6) = 10;
value.LowerBound(9:10) = 1e4;
value.UpperBound(9:10) = 1e6;
value.LowerBound(24) = -0.05;
value.UpperBound(24) = 0.05;

value.StateLossWeights = [1; 1; 3; 3; 0.5; 0.5; 0.5; 0.5];
value.CovarianceJitter = 1e-9;
value.PsdEpsilon = 1e-5;
value.MaxIterations = 26;

value.FatropTolerance = 1e-8;
value.FatropMaxIterations = 20;
value.KktTolerance = 1e-7;
value.ConstraintTolerance = 1e-8;

value.AdamLearningRate = 0.03;
value.AdamBeta1 = 0.9;
value.AdamBeta2 = 0.999;
value.ArmijoRho = 1e-4;
value.ArmijoBeta = 0.5;
value.ArmijoMaxSteps = 20;
value.TrustRegionFraction = 0.25;
end
