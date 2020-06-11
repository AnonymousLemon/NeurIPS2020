function [f,g,Hv] = cyclic6(u,X)
if nargin==0
    tic;
    runTest
    toc;
    return;
end

n = length(u)/2;
assert( (n == ceil(n)) && (n == floor(n)) );
u1 = u(1:(n)); u2 = u((n+1):end);
x1 = X(:,1); x2 = X(:,2);
assert( (length(x1) == length(u1)) && length(x2) == length(u2) );
M = 1E4;
a = -x1 + rand(n,1)/100;
b = -x1 + rand(n,1)/100;
c = -x1;
f = (u2 - x2)'*((u1 + a) + (u1 + b).*u1 + M*(u1 + c).*(u1 + c))/n;
if nargout >= 2
    g1 = (u2 - x2).*(1 + 2*u1 + b + 2*M*(u1 + c))/n;
    g2 = ((u1 + a) + (u1 + b).*u1 + M*(u1 + c).*(u1 + c))/n;
    g = [g1;
         g2];
end
if nargout == 3
    Hv = @(v) HVP(M, b, c, x2, n, u1, u2, v);
end

end

function Hv = HVP(M, b, c, x2, n, u1, u2, v)
assert( n == length(v)/2 );
v1 = v(1:(n)); v2 = v((n+1):end);
Hv1 = (u2 - x2).*(2 + 2*M).*v1 + (1 + 2*u1 + b + 2*M*(u1 + c)).*v2;
Hv2 = (1 + 2*u1 + b + 2*M*(u1 + c)).*v1;
Hv = [Hv1/n;
      Hv2/n];
end

function runTest
clc; clear all; close all; rehash;
phi=3/14; n = 14; 
X = cyclic6seq(phi, n);
derivativeTest02(@(w) cyclic6(w,X),randn(n*2,1), true)
end
