close all
clear all

params = LoadDefaultParams();

[t,sol] = CaricModelRun(params);

% Output 
Eout = sol(:,1);
hout = sol(:,2);
nout = sol(:,3);

% Plot all the variables (E,h,n)
figure()
subplot(3,1,1)
plot(t,Eout,'r','LineWidth',2);
xlabel('Time (ms)')
ylabel('Voltage (E, mV)')

subplot(3,1,2)
plot(t,hout,'b','LineWidth',2);
xlabel('Time (ms)')
ylabel('h gate (dimensionless)')

subplot(3,1,3)
plot(t,nout,'b','LineWidth',2);
xlabel('Time (ms)')
ylabel('n gate (dimensionless)')