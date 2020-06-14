function [x,hist] = minFunc(objFun,x0,args,method)


if strcmp(method, 'newton-cg') 
    [x,hist] = optNewtonCG(objFun,x0,args, method);
end
if strcmp(method, 'newton-mr')
    [x,hist] = optNewtonMR(objFun,x0,args);
end
if strcmp(method, 'trust-region')
    [x,hist] = optTrustRegion(objFun,x0,args);
end
end