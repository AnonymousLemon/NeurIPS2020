function [x,hist] = optGaussNewtonCG(objFun,x0,args)

%% Set general arguments.
if isfield(args,'maxItrs'); maxItrs = args.maxItrs; else; maxItrs = 20; end
if isfield(args,'maxProps'); maxProps = args.maxProps; else; maxProps = 1000; end
if isfield(args,'gradTol'); gradTol = args.gradTol; else; gradTol = 1E-6; end
if isfield(args,'alpha'); alpha = args.alpha; else; alpha = 1; end
if isfield(args,'testFun'); testFun = args.testFun; else; testFun = []; end
if isfield(args,'exact'); exact = args.exact; else; exact = false; end

hist.objVal = [];
hist.props = 1;
hist.norm_g = [];
hist.testVal = zeros(1,1);
hist.elapsed_time = [];
xk = x0;
k = 0;

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
    logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t %24.2f\n';
else
    %fprintf('%5s %10s %17s %13s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
    logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t %15.2f\n';
end

logBodyk_newton = '%5i  %13g %13.2e  \t%10g %12.2e %7g %12.2g%12.2g%14g%15.2f\n';
%%%%%%%%%%%%%%%%% End of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
hist.elapsed_time = 1E-16;
hist = recordHistory(hist, objFun, xk, testFun);
while true
    current_props = 0;
    current_elapsed_time = 0;
    tic;
    [fk,gk,~,Gk] = objFun(xk);
    hist.norm_g = [hist.norm_g; norm(gk)];
    if k == 0
        current_props = current_props + 2;
    else
        current_props = current_props + 1;
    end
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    if k >=1
        hist = recordHistory(hist, objFun, xk, testFun);
        assert(length(hist.elapsed_time) == length(hist.objVal));
        assert(length(hist.elapsed_time) == length(hist.testVal));
    end
    if mod(k,10) == 0
        if exact == false
            fprintf('%5s %12s %15s %15s %10s %12s %12s %12s %10s %16s\n','k','fun','norm(g)', 'time(sec)', 'alpha', 'PCGitr', 'PCG Flag', 'PCG RelRes',  'Props', 'Test Results');
        else
            
            fprintf('%5s %11s %17s %15s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
        end
    end
    if k >= 1
        if exact == false
            fprintf( logBodyk_newton, k, hist.objVal(end) , norm(gk), hist.elapsed_time(end), alpha, PCGiter, flagPCG,relresPCG,hist.props(end),hist.testVal(end) );
        else
            fprintf( logBody_other, k, hist.objVal(end), norm(gk), hist.elapsed_time(end), alpha, hist.props(end),hist.testVal(end) );
            
        end
    else
        fprintf( logBody0, k, hist.objVal(end), norm(gk), hist.testVal(end));
    end
    if norm(gk) < gradTol || k > maxItrs || hist.props(end) > maxProps
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
        pk = -Gk\gk;
    else
        [pk,flagPCG,relresPCG,PCGiter] = myPCG(Gk, -gk, subProbRelTol, subProbMaxItr);
        current_props = current_props + (1+PCGiter)*2;
        %assert(flagPCG == 0);
        if(isa(Gk,'function_handle'))
            assert(dot(pk,gk) <= -0.5*dot(pk,(Gk(pk))));
        else
            assert(dot(pk,gk) <= -0.5*dot(pk,(Gk*pk)));
        end
    end
    
    [alpha,itrLS] = lineSearchArmijo(objFun, xk, fk, gk, pk, args.alpha, args.beta1, linesearchMaxItrs);
    current_props = current_props + (itrLS+1);
    xk = xk + alpha*pk;
    g_prev = gk;
    p_prev = pk;
    alpha_prev = alpha;
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    hist.elapsed_time = [hist.elapsed_time, hist.elapsed_time(end) + current_elapsed_time];
    k = k + 1;
    hist.props = [hist.props, hist.props(end) + current_props];
end
x = xk;
end