function [x,hist] = optNewtonMR(objFun,x0,args)
global sampleSize;

%% Set arguments.
if isfield(args,'maxItrs'); maxItrs = args.maxItrs; else; maxItrs = 20; end
if isfield(args,'subProbMaxItr'); subProbMaxItr = args.subProbMaxItr; else; subProbMaxItr = 100; end
if isfield(args,'maxProps'); maxProps = args.maxProps; else; maxProps = 1000; end
if isfield(args,'gradTol'); gradTol = args.gradTol; else; gradTol = 1E-6; end
if isfield(args,'alpha'); alpha = args.alpha; else; alpha = 1; end
if isfield(args,'subProbSolver'); subProbSolver  = args.subProbSolver ; else; subProbSolver  = 'symmlq'; end
if isfield(args,'testFun'); testFun = args.testFun; else; testFun = []; end
if isfield(args,'linesearchMaxItrs'); linesearchMaxItrs = args.linesearchMaxItrs; else; linesearchMaxItrs = 100; end
if isfield(args,'subProbRelTol'); subProbRelTol = args.subProbRelTol; else; subProbRelTol = 1E-4; end



hist.objVal = [];
hist.props = 1;
hist.gradNorm = [];
hist.testVal = zeros(1,1);
xk = x0;
k = 0;

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
    [~,gk,Hk] = objFun(xk);
   
    
    if k == 0
        current_props = current_props + 2;
    end
    if k >=1
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
    if norm(gk) < gradTol || k >= maxItrs || hist.props(end) > maxProps || alpha == 0
        break;
    end
    
    if isa(Hk,'function_handle')
        
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
        
        assert( strcmp(subProbSolver, 'minres-qlp') || ...
            strcmp(subProbSolver, 'minres') || ...
            strcmp(subProbSolver, 'bicgstab') || ...
            strcmp(subProbSolver, 'symmlq') || ...
            strcmp(subProbSolver, 'sqmr') || ...
            strcmp(subProbSolver, 'cgs'));
        
        if strcmp(args.subProbSolver, 'minres-qlp')
            [pk,subProbFlag,subProbRelres,subProbIters] = myMINRESQLP(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'sqmr')
            [pk,subProbFlag,subProbRelres,subProbIters] = mySQMR(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'symmlq')
            [pk,subProbFlag,subProbRelres,subProbIters] = mySYMMLQ(Hk,-gk,subProbRelTol,subProbMaxItr,M);
            if subProbIters == 0
                fprintf(2,'---------------------------------------------------\n')
                fprintf(2,'zero iteration in SYMMLQ...Switching to MINRES-QLP.\n')
                fprintf(2,'---------------------------------------------------\n')
                args.subProbSolver = 'minres-qlp';
            end
        end
        if strcmp(args.subProbSolver, 'minres')
            [pk,subProbFlag,subProbRelres,subProbIters] = minres(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'bicgstab')
            [pk,subProbFlag,subProbRelres,subProbIters] = bicgstab(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'cgs')
            [pk,subProbFlag,subProbRelres,subProbIters] = cgs(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        Hkgk = Hk(gk);
    else
%         pk = lsqminnorm(Hk,-gk);
        pk = -pinv(Hk)*gk;
        subProbFlag = 0;
        subProbRelres = norm(Hk*pk+gk)/norm(gk);
        subProbIters = size(Hk,1);
        Hkgk = Hk*gk;
    end
    current_props = current_props + subProbIters*2*sampleSize;
    if args.Prec && k >= 1
        assert(false);
        [alpha,itrLS] = lineSearchWolfeStrong(@(x) lineSearchFun(x,objFun), xk, 0.5*n2g*n2g, Hkgk, pk, args.alpha, args.beta1, args.beta2, linesearchMaxItrs);
    else
        n2g = norm(gk);
        [alpha,itrLS] = lineSearchArmijo(@(x) lineSearchFun(x,objFun), xk, 0.5*n2g*n2g, Hkgk, pk, args.alpha, args.beta1, linesearchMaxItrs);
        %[alpha,itrLS] = lineSearchWolfeStrong(@(x) lineSearchFun(x,objFun), xk, 0.5*n2g*n2g, Hkgk, pk, args.alpha, args.beta1, args.beta2, linesearchMaxItrs);
    end
    current_props = current_props + (itrLS+1)*2;
    %     dot(pk,Hkgk)
    xk = xk + alpha*pk;
    g_prev = gk;
    p_prev = pk;
    alpha_prev = alpha;
    k = k + 1;
    hist.props = [hist.props, hist.props(end) + current_props];
end
x = xk;
end

function [fLS, gLS] = lineSearchFun(x,objFun)
if nargout == 1
    [~,g] = objFun(x);
    n2g = norm(g);
    fLS = 0.5*n2g*n2g;
else
    assert(nargout == 2)
    [~,g, H] = objFun(x);
    n2g = norm(g);
    fLS = 0.5*n2g*n2g;
    gLS = H(g);
end
end