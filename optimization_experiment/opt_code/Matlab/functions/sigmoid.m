function [f,g,H] = sigmoid(t)
if nargin==0
    tic;
    runTest
    toc;
    return;
end

f = 1./(exp(-t) + 1);

if nargout >= 2
    g = ( 1 ./ (exp(-t) + 1) ) .* ( 1 ./ (exp(t) + 1) );
end

if nargout >= 3
    H = zeros(length(t),1);
    idx = (t >= 0);
    tp = t(idx);
    H(idx) = -(1 - exp(-tp)).* 1./(exp(-tp) + 1) .* 1./(exp(-tp) + 1) .* 1./(exp(tp) + 1);
    idx = (t < 0);
    tm = t(idx);
    H(idx) = -((exp(tm).*(exp(tm) - 1)))./((exp(tm) + 1).^3);
end

end

function runTest
clc; clear all; close all; rehash;
derivativeTestMVP(@(x) sigmoid(x),randn)
end


% f = 1./(exp(-t) + 1);
%
% if nargout >= 2
%     g = exp(-t) ./ (exp(-t) + 1).^2;
% end
%
% if nargout >= 3
%  H = -((exp(t).*(exp(t) - 1)))./((exp(t) + 1).^3);
%end