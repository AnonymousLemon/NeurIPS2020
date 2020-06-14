function [f,g,H] = tanhFun(t,c)
if nargin==0
    tic;
    runTest
    toc;
    return;
end

f = 0.5*(1+tanh(c*t));

if nargout >= 2
    g = 0.5*c*(1 - (tanh(c*t)).^2);
end

if nargout >= 3
    H = ( c*c*tanh(c*t) ) .* (( tanh(c*t) ).^2 - 1);
end

end

function runTest
clc; clear all; close all; rehash;
derivativeTest(@(x) tanhFun(x,1),randn)
end
