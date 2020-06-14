function [x,hist] = optNewtonCG(objFun,x0,args,method)

global sampleSize;

assert( strcmp(method, 'newton-cg'));

%% Set general arguments.
if isfield(args,'maxItrs'); maxItrs = args.maxItrs; else; maxItrs = 20; end
if isfield(args,'subProbMaxItr'); subProbMaxItr = args.subProbMaxItr; else; subProbMaxItr = 100; end
if isfield(args,'maxProps'); maxProps = args.maxProps; else; maxProps = 1000; end
if isfield(args,'gradTol'); gradTol = args.gradTol; else; gradTol = 1E-6; end
if isfield(args,'alpha'); alpha = args.alpha; else; alpha = 1; end
if isfield(args,'testFun'); testFun = args.testFun; else; testFun = []; end
if isfield(args,'linesearchMaxItrs'); linesearchMaxItrs = args.linesearchMaxItrs; else; linesearchMaxItrs = 100; end
if isfield(args,'subProbRelTol'); subProbRelTol = args.subProbRelTol; else; subProbRelTol = 1E-4; end
if isfield(args,'exact'); exact = args.exact; else; exact = false; end


hist.objVal = [];
hist.props = 1;
hist.gradNorm = [];
hist.testVal = zeros(1,1);
xk = x0;
k = 0;
flagPCG = 0;
%% Get the arguments specific to each method

if args.Prec
    S = [];
    Y = [];
end

%%%%%%%%%%%%%%%%% Start of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
logBody0 = '%5i  %13g %13.2e \t\t\t\t %30.2f\n';
logBodyk = '%5i  %13g %13.2e %13.2e %16.2e %13.2f\n';
%%%%%%%%%%%%%%%%% End of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
hist = recordHistory(hist, objFun, xk, testFun);
while true
    current_props = 0;
    if strcmp(method, 'newton-cg')
        [fk,gk,Hk] = objFun(xk);
        Ak = Hk;
    else
        assert(strcmp(method, 'gauss-newton'));
        [fk,gk,~,Gk] = objFun(xk);
        Ak = Gk;
    end
    if k == 0
        current_props = current_props + 2;
    else
        current_props = current_props + 1;
    end
    if k >=1 && flagPCG ~= -1
        hist = recordHistory(hist, objFun, xk, testFun);
    end
    if mod(k,10) == 0
        fprintf('%5s %11s %16s %10s %15s %18s\n','k','fun','norm(g)', 'alpha', 'Props', 'Test Results');
    end
    if k >= 1
        fprintf( logBodyk, k, hist.objVal(end) , norm(gk), alpha,hist.props(end),hist.testVal(end) );
    else
        fprintf( logBody0, k, hist.objVal(end), norm(gk), hist.testVal(end));
    end
    if norm(gk) < gradTol || k >= maxItrs || hist.props(end) > maxProps || alpha == 0 || flagPCG == -1
        break;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% Precond %%%%%%%%%%%%%%%%%%%%%%%%
    if args.Prec && k >= 1
        s = alpha_prev*p_prev;
        y = gk - g_prev;
        L = 20;
        if size(S,2) >=  L
            S = [S(:,2:end),s];
            Y = [Y(:,2:end),y];
        else
            S = [S, s];
            Y = [Y, y];
        end
        M = @(v) lbfgs(v,S,Y);
    else
        M = @(v) v;
    end
    %%%%%%%%%%%%%%%%%%%%%%%% Precond %%%%%%%%%%%%%%%%%%%%%%%%
    
    if ~isa(Ak,'function_handle') %if exact == true
        exact = true;
        assert(~isa(Ak,'function_handle'));
        pk = -Ak\gk;
        subProbFlag = 0;
        subProbRelres = norm(Ak*pk+gk)/norm(gk);
        subProbIters = size(Ak,1);
    else
        exact = false;
        [pk,flagPCG,relresPCG,PCGiter] = myPCG(Ak, -gk, subProbRelTol, subProbMaxItr,M);
        if flagPCG == -1
            warning('Newton CG: PCG failed...exiting the loop.....iter: %g\n',k);
            linesearchMaxItrs = 0; % so we don't do line-search
        end
        %current_props = current_props + (1+PCGiter)*2;
        %assert(flagPCG == 0);
        if(isa(Ak,'function_handle'))
            assert(dot(pk,gk) <= -0.5*dot(pk,(Ak(pk))));
        else
            assert(dot(pk,gk) <= -0.5*dot(pk,(Ak*pk)));
        end
    end
    current_props = current_props + subProbIters*2*sampleSize;
    
    [alpha,itrLS] = lineSearchArmijo(objFun, xk, fk, gk, pk, args.alpha, args.beta1, linesearchMaxItrs);
    current_props = current_props + (itrLS+1);
    xk = xk + alpha*pk;
    g_prev = gk;
    p_prev = pk;
    alpha_prev = alpha;
    if flagPCG ~= -1
        hist.props = [hist.props, hist.props(end) + current_props];
    end
    k = k + 1;
end
x = xk;
end