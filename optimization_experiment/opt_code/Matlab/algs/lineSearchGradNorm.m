function [alpha,itrLS] = lineSearchGradNorm(objFun, xk, gk, Hkgk, pk, alpha, beta, linesearchMaxItrs)
[~,g_alpha] = objFun(xk+alpha*pk);
itrLS = 0;
while( ( ( ( norm(g_alpha) )^2 )  >  ( ( ( norm(gk) )^2 ) + 2*alpha*beta*dot(pk, Hkgk) )  )  && itrLS < linesearchMaxItrs )
    alpha = alpha/2;
    [~,g_alpha] = objFun(xk+alpha*pk);
    itrLS = itrLS + 1;
end
end

