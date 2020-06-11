function [f,g,Hv,H] = regularizerNonConvex(x, lambda)
if nargin==0
    tic;
    runTest
    toc;
    return;
end
a = 1;
x2 = x.^2;
f = lambda*sum((a*x2)./(1+a*x2));
if nargout >= 2
    g = lambda*(2.*a.*x)./(a.*x2 + 1).^2;
end
if nargout >= 3
    Hv = @(v) lambda*((-(2.*a.*(3.*a.*x2 - 1))./(a.*x2 + 1).^3)).*v;
end
if nargout ==4
    H = lambda*diag((-(2.*a.*(3.*a.*x2 - 1))./(a.*x2 + 1).^3));
end 
end

function runTest
clc; clear all; close all; rehash;
d = 100;
lambda = 1;
regularizer = @(w) regularizerNonConvex(w, lambda);
derivativeTest(regularizer,randn(d,1))
end

