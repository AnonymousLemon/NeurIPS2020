function [alpha,itrLS] = lineSearchArmijo(objFun, xk, fk, gk, pk, alpha, beta, linesearchMaxItrs)
f_alpha = objFun(xk+alpha*pk);
itrLS = 0;
while( ( f_alpha > ( fk + alpha*beta*dot(gk,pk) ) ) && itrLS < linesearchMaxItrs )
    alpha = alpha/2;
    f_alpha = objFun(xk+alpha*pk);
    itrLS = itrLS + 1;
end
end
