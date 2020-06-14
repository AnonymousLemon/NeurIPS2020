function [f,g,H] = invexFunReg02(x,A, regFun)
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
if exist('regularizer','var') && ~isempty(regFun)
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
f = sum((Ax + (6.*cos(Ax))./5).*(Ax + (7.*sin(Ax))./5))/n + 1 + f_reg;

if nargout >= 2
    g = A'*((Ax + (6.*cos(Ax))./5).*((7.*cos(Ax))./5 + 1) - (Ax + (7.*sin(Ax))./5).*((6.*sin(Ax))./5 - 1))/n + g_reg;
end

if nargout >= 3
    H = @(v) A'*( ( -(7.*sin(Ax).*(Ax + (6.*cos(Ax))./5))./5 - (6.*cos(Ax).*(Ax + (7.*sin(Ax))./5))./5 - 2.*((7.*cos(Ax))./5 + 1).*((6.*sin(Ax))./5 - 1) ) .* (A*v) )/n + Hv_reg(v);
end

end

function runTest
clc; clear all; close all; rehash;
p = 10000;
n = 100;
A = randn(n,p);
derivativeTest(@(x) invexFun02(x,A),randn(p,1))
end

