function [x,hist] = optLBFGS(objFun,x0,args)
%% Set general arguments.
if isfield(args,'maxItrs'); maxItrs = args.maxItrs; else; maxItrs = 20; end
if isfield(args,'maxProps'); maxProps = args.maxProps; else; maxProps = 1000; end
if isfield(args,'gradTol'); gradTol = args.gradTol; else; gradTol = 1E-6; end
if isfield(args,'alpha'); alpha = args.alpha; else; alpha = 1; end
if isfield(args,'testFun'); testFun = args.testFun; else; testFun = []; end
hist.objVal = [];
hist.props = 1;
hist.gradNorm = [];
hist.testVal = zeros(1,1);
hist.elapsed_time = [];
xk = x0;
k = 0;

linesearchMaxItrs = args.linesearchMaxItrs ;
L = args.L;
S = [];
Y = [];
%%%%%%%%%%%%%%%%% Start of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
%fprintf('%5s %10s %17s %13s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t %15.2f\n';
logBody_other = '%5i  %13g %13.2e %14g %11.2e %6g%12.2f\n';
%%%%%%%%%%%%%%%%% End of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
hist.elapsed_time = 1E-16;
hist = recordHistory(hist, objFun, xk, testFun);
while true
    current_props = 0;
    current_elapsed_time = 0;
    tic;
    [fk,gk] = objFun(xk);
    if k == 0
        current_props = current_props + 2;
    end
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    if k >=1
        hist = recordHistory(hist, objFun, xk, testFun);
        assert(length(hist.elapsed_time) == length(hist.objVal));
        assert(length(hist.elapsed_time) == length(hist.testVal));
    end
    if mod(k,10) == 0
        fprintf('%5s %11s %17s %15s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
    end
    if k >= 1
        fprintf( logBody_other, k, hist.objVal(end), norm(gk), hist.elapsed_time(end), alpha, hist.props(end),hist.testVal(end) );
    else
        fprintf( logBody0, k, hist.objVal(end), norm(gk), hist.testVal(end));
    end
    if norm(gk) < gradTol || k >= maxItrs || hist.props(end) > maxProps || alpha == 0
        break;
    end
    tic;
    if k == 0
        pk = -gk;
        Y = [S, zeros(length(gk),0)];
        S = [S , zeros(length(gk),0)];
    else
        s = alpha_prev*p_prev;
        y = gk - g_prev;
        
        if size(S,2) >=  L
            S = [S(:,2:end),s];
            Y = [Y(:,2:end),y];
        else
            S = [S, s];
            Y = [Y, y];
        end
        pk = -lbfgs(gk,S,Y);
    end
    
    [alpha,itrLS] = lineSearchWolfeStrong(objFun, xk, fk, gk, pk, args.alpha, args.beta1, args.beta2, linesearchMaxItrs);
    
    current_props = current_props + (itrLS+1)*2;
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