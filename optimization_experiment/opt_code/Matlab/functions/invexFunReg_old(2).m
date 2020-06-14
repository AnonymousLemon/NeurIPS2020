function [f,g,H] = invexFunReg(x,A, regFun)
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
f = sum(3.*sin(Ax).^2 + Ax.^2)/n + f_reg;

if nargout >= 2
    g = A'*(2.*Ax + 3.*sin(2.*Ax))/n + g_reg;
end

if nargout >= 3
    H = @(v) A'*( (8 - 12.*sin(Ax).^2) .* (A*v) )/n + Hv_reg(v);
end

end

function runTest
clc; clear all; close all; rehash;
p = 100;
n = 10;
A = randn(n,p);
derivativeTest(@(x) invexFun(x,A),randn(p,1))
end

