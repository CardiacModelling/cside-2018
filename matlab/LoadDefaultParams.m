
function params = LoadDefaultParams()

%%%%%%%% These are the default parameter values. Change for parameter estimation. 

k1 = 3.0/40.0;    % p1
k2 = 1.0/25.0;    % p2
k3 = 1.0/10.0;    % p3
E1 = -280.0/3.0;  % p4 = -93.3333
Ena = 40.0;       % p5
Edag = -80.0;     % p6
Estar = -15.0;    % p7
Fh = 1.0/2.0;     % p8
fn = 1.0/270.0;   % p9
Gna = 100.0/3.0;  % p10
g21 = -2.0;       % p11
g22 = -9.0;       % p12

% %% Override a couple from the spline fit exercise
Edag = -52;
Fh = 0.3541;
fn = 0.0147;
k2 = 0.016; % Guess from 1D sensitivity

%%% Ensure the parameters are put in the "params" vector in this order (so CariqEq.m gets the correct parameter values)

params = [k1,k2,k3,E1,Ena,Edag,Estar,Fh,fn,Gna,g21,g22]';