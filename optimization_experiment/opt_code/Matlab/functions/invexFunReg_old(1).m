function [f,g,H] = invexFunReg03(x,A, regFun)
if nargin==0
    tic;
    runTest
    toc;
    return;
end
n = size(A,1);
f_reg = 0;
g_reg = 0;
Hv_reg = @(v) 0;
if exist('regFun','var') && ~isempty(regFun)
    assert(isa(regFun,'function_handle'));
    if nargout >= 3
        [f_reg,g_reg,Hv_reg] = regFun(x);
    end
    if nargout == 2
        [f_reg,g_reg] = regFun(x);
    end
    if nargout == 1
        f_reg = regFun(x);
    end
end

Ax = A*x;
idx1 = Ax >= 1;
idx2 = (Ax >= -1) & (Ax < 1);
idx3 = Ax < -1;
assert(sum(idx1 & idx2) == 0);
assert(sum(idx1 & idx3) == 0);
assert(sum(idx2 & idx3) == 0);
Ax1 = Ax(idx1);
Ax2 = Ax(idx2);
Ax3 = Ax(idx3);
f1 = 0.5*Ax1.^2;
f2 = -0.5*Ax2.^2 + 2*Ax2 - 1;
f3 = 0.5*Ax3.^2 + 4*Ax3;
f = ( sum(f1) + sum(f2) + sum(f3) )/n + f_reg + 10;

if nargout >= 2
    A1 = A(idx1,:);
    A2 = A(idx2,:);
    A3 = A(idx3,:);
    g1 = 0; g2 = 0; g3 = 0; 
    c = ones(size(A,1),1);
    if ~isempty(A1)
        g1 = A1'*Ax1;
    end
    if ~isempty(A2)
        g2 = A2'*(2-Ax2);
    end
    if ~isempty(A3)
        g3 = A3'*(Ax3 + 4);
    end
    g = ( g1 + g2 + g3 ) / n + g_reg;
end

if nargout >= 3
    H = @(v) A'*( ( idx1 + idx3 - idx2 ) .* (A*v) )/n + Hv_reg(v);
end

end

function runTest
clc; clear all; close all; rehash;
p = 100;
n = 10;
A = randn(n,p);
A = eye(p,p);
x = [-10:0.1:10]';
for i = 1:length(x)
    [f(i),g(i),Hv] = invexFunReg03(x(i),1);
    H(i) = Hv(1);
end
figure(1);
plot(x,f);
figure(2);
plot(x,g);
figure(3);
plot(x,H);
plot(x,g.*H);
%derivativeTest(@(x) invexFunReg03(x,A),10*randn(p,1))
end

