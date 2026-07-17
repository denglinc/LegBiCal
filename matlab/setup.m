function setup()
%SETUP Add the MATLAB package and CasADi/Fatrop to the search path.

root = fileparts(mfilename('fullpath'));
addpath(root);
casadiPath = getenv('CASADI_MATLAB_PATH');
if ~isempty(casadiPath)
    addpath(casadiPath);
end
if exist('casadi.Opti', 'class') ~= 8 || ~casadi.has_nlpsol('fatrop')
    error('legbical:Casadi', ...
        'Set CASADI_MATLAB_PATH to a CasADi build that includes Fatrop.');
end
end
