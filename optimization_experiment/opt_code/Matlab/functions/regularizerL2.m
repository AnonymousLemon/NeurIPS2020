function [f,g,Hv,H] = regularizerL2(x, lambda, D)
if nargin==0
    tic;
    runTest
    toc;
    return;
end
assert(nargin >= 2, 'Not enough input given for the regularizerL2 function.');
if nargin == 2
    D = speye(length(x));
end
f = 0.5*lambda*(norm(D*x))^2;
if nargout >= 2
    g = lambda*(D'*(D*x));
end
if nargout >= 3
    Hv = @(v) lambda*(D'*(D*v));
end
if nargout ==4
    H = lambda*(D'*D);
end
end

function runTest
clc; clear all; close all; rehash;
d = 100;
D = randn(d,d);
lambda = 1;
regularizer = @(w) regularizerL2(w, lambda);
derivativeTest(regularizer,randn(d,1))
end

