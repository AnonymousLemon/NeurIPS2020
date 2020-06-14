function [x,flagCG,relresCG,iterCG] = myPCG(A,b,tol,maxiter,P)
tol2 = tol^2;
x = zeros(size(b));
r = b - A(x);
if not(exist('P','var'))
    %P = @(x) x;
    P = speye(length(x));
end
verbose = 0;

h = Precond(P,r);
delta = r'*h;
bb = (b')*(Precond(P,b));
p = h;
iter = 0;
x_best = x;%xs = zeros(length(x),maxiter);
best_rel_residual = inf;
flagCG = 0;
while ( delta > tol2 * bb && iter < maxiter && norm(r)/norm(b) > tol)
    Ap = A(p);
    pAp = (p'*Ap);
    if pAp <= 0
       if abs(pAp) < 1E-32
           warning('PCG: pAp is too small!')
           break;
       end
       warning('PCG: The matrix is not positive definite.....iter: %g, pAp: %g\n',iter, pAp);
       flagCG = -1;
       break;
    end
    alfa = delta /(p'*Ap);
    x = x + alfa*p;
    r = r - alfa*Ap;
    rel_res_k = norm(r)/norm(b);
    if best_rel_residual > rel_res_k
        x_best = x;
        best_rel_residual = rel_res_k;
    end
    if verbose && mod(iter,100) == 0
        fprintf('Iter: %i, My PCG Rel Residual: %g\n',iter, rel_res_k);
    end
    h = Precond(P,r);
    prev_delta = delta;
    delta = r'*h;
    p = h + (delta/prev_delta)*p;
    iter = iter + 1;
    %xs(:,iter) = x;
end
% xs = xs(:,1:iter);
% x = xs(:,best_iter);
x = x_best;
%relresCG = norm(A(x) - b)/norm(b);
relresCG = best_rel_residual;
iterCG = iter;
if flagCG == 0
    flagCG = (iter >= maxiter);
end
end

function y = Precond(P,r)
if isa(P,'function_handle')
    y = P(r);
%     assert(false);
%     [y,~] = pcg(P,r,1E-12,1000);
else
    y = P\r;
end
end