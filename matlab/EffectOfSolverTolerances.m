close all
clear all

dataset = load('data/noisy_data_1.tsv');

params = LoadDefaultParams();
param_scalings = ones(12,1);
sigmas = [15.0; 0.15; 0.1]; % standard deviations on V,h,n


for t=3:5

    solver_options = odeset('RelTol', 10.^-t);
    
    N=1000.0;
    param_val = zeros(N,1);
    logL_val = zeros(N,1);

    for i=1:N
        param_scalings(1) = (0.9+0.2*i./N);
        param_val(i) = params(1)*param_scalings(1);

        logL = EvaluateLogLikelihood(dataset, sigmas, param_scalings, solver_options);
        logL_val(i) = logL;
    end

    plot(param_val, logL_val)
    xlabel('Parameter 1')
    ylabel('LogLikelihood')
    hold all

end
legend({'RelTol = 1e-3 (default)','RelTol = 1e-4','RelTol = 1e-5'},'Location','SouthWest')
