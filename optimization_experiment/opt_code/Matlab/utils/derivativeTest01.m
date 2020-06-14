function derivativeTest01(fun,x0)
[f0,g0,H0v,H0] = fun(x0);
dx = randn(size(x0));
M = 30;
dxs = zeros(M,1);
firstOrderError = zeros(M,1);
Order_1st = zeros(M-1,1);
secondOrderError = zeros(M,1);
Order_2nd = zeros(M-1,1);
secondOrderErrorMVP = zeros(M,1);
Order_2ndMVP = zeros(M-1,1);
for i = 1:M
    x = x0 + dx;
    [f,~] = fun(x);
    firstOrderError(i) = abs(f - ( f0 + (dx')*g0 ))/abs(f0);
    secondOrderError(i) = abs(f - ( f0 + (dx')*g0 + 0.5*(dx')*(H0*dx)))/abs(f0);
    secondOrderErrorMVP(i) = abs(f - ( f0 + (dx')*g0 + 0.5*(dx')*(H0v(dx))))/abs(f0);
    fprintf('First Order Error: %g, Second Order Error Explicit: %g, Second Order Error MVP: %g\n',firstOrderError(i),secondOrderError(i),secondOrderErrorMVP(i));
    if i>1
        Order_1st(i-1) = log2(firstOrderError(i-1)/firstOrderError(i));
        Order_2nd(i-1) = log2(secondOrderError(i-1)/secondOrderError(i));
        Order_2ndMVP(i-1) = log2(secondOrderErrorMVP(i-1)/secondOrderErrorMVP(i));
    end
    dxs(i) = norm(dx);
    dx = dx/2;
end

figure(1);
step = 2.^-(1:M);
loglog(step,abs(firstOrderError),'r',step,dxs.^2,'b');
legend('1st Order Error','order');
set(gca,'XDir','reverse')

figure(2);
semilogx(step(2:end),Order_1st,'r');
legend('1st Order');
set(gca,'XDir','reverse')

figure(3);
loglog(step,abs(secondOrderErrorMVP),'k--',step,dxs.^3,'b');
legend('2nd Order Error - MVP','order');
set(gca,'XDir','reverse')

figure(4);
semilogx(step(2:end),Order_2nd, 'r', step(2:end),Order_2ndMVP, 'k');
legend('2nd Order - Explicit','2nd Order - MVP');
set(gca,'XDir','reverse')

autoArrangeFigures(2,2,1);
fprintf('press any key if you want to proceed\n')
pause
clc; clear all; close all; rehash;

end