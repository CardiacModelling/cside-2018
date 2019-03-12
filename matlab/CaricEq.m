function ydot = CaricEq(t,x,params)
  
k1 = params(1); 
k2 = params(2); 
k3 = params(3); 
E1 = params(4); 
Ena = params(5); 
Edag = params(6); 
Estar = params(7); 
Fh = params(8); 
fn = params(9); 
Gna = params(10); 
g21 = params(11); 
g22 = params(12);

E2 = ((k1/k2)+1)*Edag - E1*(k1/k2);
E3 = ((k2/k3)+1)*Estar - E2*(k2/k3);

%% Gary edited the equations to hardcode these (take out calculations).
%r = 1; % do not change
%eps1 = 1; % do not change
%eps2 = 1; % do not change
%%

if x(1) < Edag
    G = k1*(E1-x(1));
elseif (x(1)>=Edag) && (x(1) < Estar)
    G = k2*(x(1)-E2);
else x(1) >= Estar;
    G = k3*(E3-x(1));
end

  ydot = [Gna*(Ena-x(1))*heaviside(x(1)-Estar)*x(2)+ ...
      ((g21*heaviside(Edag-x(1))+g22*heaviside(x(1)-Edag))*x(3)^4+G); 
    Fh*(heaviside(Edag-x(1))-x(2)); ...
    fn*(heaviside(x(1)-Edag)-x(3))];

if (max(ydot)>1e6)
    warning('YIKES 1!')
end

if (min(ydot)<-1e6)
    warning('YIKES 2!')
end
end
