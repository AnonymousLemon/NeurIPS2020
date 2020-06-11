function [x,hist] = optNewtonCG(objFun,x0,args,method)

assert( strcmp(method, 'newton-cg') || ...
    strcmp(method, 'gauss-newton'));

%% Set general arguments.
if isfield(args,'maxItrs'); maxItrs = args.maxItrs; else; maxItrs = 20; end
if isfield(args,'maxProps'); maxProps = args.maxProps; else; maxProps = 1000; end
if isfield(args,'gradTol'); gradTol = args.gradTol; else; gradTol = 1E-6; end
if isfield(args,'alpha'); alpha = args.alpha; else; alpha = 1; end
if isfield(args,'testFun'); testFun = args.testFun; else; testFun = []; end
if isfield(args,'exact'); exact = args.exact; else; exact = false; end

hist.objVal = [];
hist.props = 1;
hist.gradNorm = [];
hist.testVal = zeros(1,1);
hist.elapsed_time = [];
xk = x0;
k = 0;
flagPCG = 0;
%% Get the arguments specific to each method
linesearchMaxItrs = args.linesearchMaxItrs ;
subProbRelTol = args.subProbRelTol;
subProbMaxItr = args.subProbMaxItr;

if args.Prec
    S = [];
    Y = [];
end

%%%%%%%%%%%%%%%%% Start of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
if exact == false
    %fprintf('%5s %12s %15s %13s %8s %10s %12s %12s %12s %16s\n','k','fun','norm(g)', 'time(sec)', 'alpha', 'PCGitr', 'PCG Flag', 'PCG RelRes',  'Props', 'Test Results');
    logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t %22.2f\n';
else
    %fprintf('%5s %10s %17s %13s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
    logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t %15.2f\n';
end

logBodyk = '%5i  %13g %13.2e  %13g %12.2e %8i %12g %13.2g %10g %13.2f\n';
%%%%%%%%%%%%%%%%% End of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
hist.elapsed_time = 1E-16;
hist = recordHistory(hist, objFun, xk, testFun);
while true
    current_props = 0;
    current_elapsed_time = 0;
    tic;
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
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    if k >=1 && flagPCG ~= -1
        hist = recordHistory(hist, objFun, xk, testFun);
        assert(length(hist.elapsed_time) == length(hist.objVal));
        assert(length(hist.elapsed_time) == length(hist.testVal));
    end
    if mod(k,10) == 0
        if exact == false
            fprintf('%5s %12s %15s %15s %10s %11s %12s %12s %10s %16s\n','k','fun','norm(g)', 'time(sec)', 'alpha', 'PCGitr', 'PCG Flag', 'PCG RelRes',  'Props', 'Test Results');
        else
            
            fprintf('%5s %11s %17s %15s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
        end
    end
    if k >= 1
        if exact == false
            fprintf( logBodyk, k, hist.objVal(end) , norm(gk), hist.elapsed_time(end), alpha, PCGiter, flagPCG,relresPCG,hist.props(end),hist.testVal(end) );
        else
            fprintf( logBody_other, k, hist.objVal(end), norm(gk), hist.elapsed_time(end), alpha, hist.props(end),hist.testVal(end) );
            
        end
    else
        fprintf( logBody0, k, hist.objVal(end), norm(gk), hist.testVal(end));
    end
    if norm(gk) < gradTol || k >= maxItrs || hist.props(end) > maxProps || flagPCG == -1
        break;
    end
    tic;
    
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
    
    if exact == true
        assert(false); % For now we are disabling the exact case
        assert(~isa(Ak,'function_handle'));
        pk = -Ak\gk;
    else
        [pk,flagPCG,relresPCG,PCGiter] = myPCG(Ak, -gk, subProbRelTol, subProbMaxItr,M);
        if flagPCG == -1
            warning('Newton CG: PCG failed...exiting the loop.....iter: %g\n',k);
            linesearchMaxItrs = 0; % so we don't do line-search
        end
        current_props = current_props + (1+PCGiter)*2;
        %assert(flagPCG == 0);
        if(isa(Ak,'function_handle'))
            assert(dot(pk,gk) <= -0.5*dot(pk,(Ak(pk))));
        else
            assert(dot(pk,gk) <= -0.5*dot(pk,(Ak*pk)));
        end
    end
    
    [alpha,itrLS] = lineSearchArmijo(objFun, xk, fk, gk, pk, args.alpha, args.beta1, linesearchMaxItrs);
    current_props = current_props + (itrLS+1);
    xk = xk + alpha*pk;
    g_prev = gk;
    p_prev = pk;
    alpha_prev = alpha;
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    if flagPCG ~= -1
        hist.elapsed_time = [hist.elapsed_time, hist.elapsed_time(end) + current_elapsed_time];
        hist.props = [hist.props, hist.props(end) + current_props];
    end
    k = k + 1;
end
x = xk;
end