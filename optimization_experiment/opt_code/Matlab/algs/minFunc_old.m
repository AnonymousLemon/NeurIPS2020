function [x,hist] = minFunc_old(objFun,x0,args,method)

assert( strcmp(method, 'newton-cg') || ...
    strcmp(method, 'gauss-newton') || ...
    strcmp(method,'newton-qlp') || ...
    strcmp(method, 'lbfgs') || ...
    strcmp(method, 'gd') || ...
    strcmp(method, 'acc_gd') ||...
    strcmp(method, 'adam') || ...
    strcmp(method, 'adadelta'));

%% Set general arguments.
if isfield(args,'maxItrs'); maxItrs = args.maxItrs; else; maxItrs = 20; end
if isfield(args,'maxProps'); maxProps = args.maxProps; else; maxProps = 1000; end
if isfield(args,'gradTol'); gradTol = args.gradTol; else; gradTol = 1E-6; end
if isfield(args,'alpha'); alpha = args.alpha; else; alpha = 1; end
if isfield(args,'testFun'); testFun = args.testFun; else; testFun = []; end
hist.objVal = [];
hist.props = 1;
hist.norm_g = [];
hist.testVal = zeros(1,1);
hist.elapsed_time = [];
xk = x0;
k = 0;

%% Get the arguments specific to each method
if strcmp(method, 'newton-cg') || strcmp(method, 'gauss-newton')
    linesearchMaxItrs = args.linesearchMaxItrs ;
    subProbRelTol = args.subProbRelTol;
    subProbMaxItr = args.subProbMaxItr;
    if isfield(args,'exact'); exact = args.exact; else; exact = false; end
end
if strcmp(method, 'newton-qlp')
    linesearchMaxItrs = args.linesearchMaxItrs ;
    subProbRelTol = args.subProbRelTol;
    subProbMaxItr = args.subProbMaxItr;
end
if strcmp(method,'lbfgs') % BFGS
    linesearchMaxItrs = args.linesearchMaxItrs ;
    L = args.L;
    S = [];
    Y = [];
end
if strcmp(method,'adam')
    beta1 = args.beta1;
    beta2 = args.beta2;
    learning_rate = args.learning_rate;
    epsilon = args.epsilon;
    m_last = zeros(size(x0));
    v_last = zeros(size(x0));
end
if strcmp(method, 'adadelta')
    beta1 = args.beta;
    learning_rate = args.learning_rate;
    epsilon = args.epsilon;
    avg_gk = zeros(size(x0)); % accumulated gradients
    avg_pk = zeros(size(x0)); % accumulated updates
end

if args.Prec 
    S = [];
    Y = [];
end
%%%%%%%%%%%%%%%%% Start of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
if ( strcmp(method,'newton-cg') || strcmp(method,'gauss-newton') ) && exact == false
    %fprintf('%5s %12s %15s %13s %8s %10s %12s %12s %12s %16s\n','k','fun','norm(g)', 'time(sec)', 'alpha', 'PCGitr', 'PCG Flag', 'PCG RelRes',  'Props', 'Test Results');
    logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t %24.2f\n';
else
    if strcmp(method,'newton-qlp')
        %fprintf('%5s %12s %15s %13s %8s %10s %12s %12s %12s %16s\n','k','fun','norm(g)', 'time(sec)', 'alpha', 'MQLPitr', 'MQLP Flag', 'MQLP RelRes',  'Props', 'Test Results');
        logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t %24.2f\n';
    else
        %fprintf('%5s %10s %17s %13s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
        logBody0 = '%5i  %13g %13.2e \t\t\t\t\t\t\t\t %15.2f\n';
    end
end
logBodyk_newton = '%5i  %13g %13.2e  \t%10g %12.2e %7g %12.2g%12.2g%14g%15.2f\n';
logBody_other = '%5i  %13g %13.2e %14g %11.2e %6g%12.2f\n';
%%%%%%%%%%%%%%%%% End of Printing Headers %%%%%%%%%%%%%%%%%%%%%%%%%%
hist.elapsed_time = 1E-16;
hist = recordHistory(hist, objFun, xk, testFun);
while true
    current_props = 0;
    current_elapsed_time = 0;
    tic;
    if strcmp(method,'newton-cg') || strcmp(method,'newton-qlp')
        [fk,gk,Hk] = objFun(xk);
    end
    if strcmp(method,'gauss-newton')
        [fk,gk,~,Gk] = objFun(xk);
    end
    if strcmp(method,'grad') || strcmp(method,'lbfgs') || strcmp(method, 'acc_grad') || strcmp(method,'adam') || strcmp(method,'adadelta')
        [fk,gk] = objFun(xk);
    end
    hist.norm_g = [hist.norm_g; norm(gk)];
    if k == 0
        current_props = current_props + 2;
    else
        if strcmp(method,'newton-cg') || strcmp(method,'gauss-newton')
            current_props = current_props + 1;
        end
        if strcmp(method,'grad') || strcmp(method, 'acc_grad') || strcmp(method,'adam') || strcmp(method,'adadelta')
            current_props = current_props + 2;
        end
    end
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    if k >=1
        hist = recordHistory(hist, objFun, xk, testFun);
        assert(length(hist.elapsed_time) == length(hist.objVal));
        assert(length(hist.elapsed_time) == length(hist.testVal));
    end
    if mod(k,10) == 0
        if ( strcmp(method,'newton-cg') || strcmp(method,'gauss-newton') ) && exact == false
            fprintf('%5s %12s %15s %15s %10s %12s %12s %12s %10s %16s\n','k','fun','norm(g)', 'time(sec)', 'alpha', 'PCGitr', 'PCG Flag', 'PCG RelRes',  'Props', 'Test Results');
        else
            if strcmp(method,'newton-qlp')
                fprintf('%5s %12s %15s %15s %10s %12s %12s %12s %10s %16s\n','k','fun','norm(g)', 'time(sec)', 'alpha', 'MQLPitr', 'MQLP Flag', 'MQLP RelRes',  'Props', 'Test Results');
            else
                fprintf('%5s %11s %17s %15s %8s %8s %8s\n','k','fun','norm(g)',  'time(sec)', 'alpha', 'Props', '  Test Results');
            end
        end
    end
    if k >= 1
        if ( strcmp(method,'newton-cg') || strcmp(method,'gauss-newton') ) && exact == false
            fprintf( logBodyk_newton, k, hist.objVal(end) , norm(gk), hist.elapsed_time(end), alpha, PCGiter, flagPCG,relresPCG,hist.props(end),hist.testVal(end) );
        else
            
            if strcmp(method,'newton-qlp')
                fprintf( logBodyk_newton, k, hist.objVal(end) , norm(gk), hist.elapsed_time(end), alpha, MINRESQLPiter, flagMINRESQLP,relresMINRESQLP,hist.props(end),hist.testVal(end) );
            else
                fprintf( logBody_other, k, hist.objVal(end), norm(gk), hist.elapsed_time(end), alpha, hist.props(end),hist.testVal(end) );
            end
        end
    else
        fprintf( logBody0, k, hist.objVal(end), norm(gk), hist.testVal(end));
    end
    if norm(gk) < gradTol || k > maxItrs || hist.props(end) > maxProps
        break;
    end
    tic;
    if strcmp(method,'grad')
        xk = xk - args.alpha*gk;
    end
    if strcmp(method,'acc_grad')
        Q = args.Q;
        if k == 0
            ys = xk;
        end
        ys1 = xk -args.alpha*gk;
        xk = (1+(sqrt(Q)-1)/(sqrt(Q)+1))*ys1 - ((sqrt(Q)-1)/(sqrt(Q)+1))*ys;
        ys = ys1;
    end
    if strcmp(method,'adam')
        % As implemented in TensorFlow: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        alpha = learning_rate * sqrt( 1 - ( beta2^(k+1) ) ) / ( 1 - ( beta1^(k+1) ) );
        m = beta1*m_last + (1 - beta1)*gk;
        v = beta2*v_last + (1 - beta2)*(gk.^2);
        xk = xk - alpha * ( m ./ ( sqrt(v) + epsilon ) );
        m_last = m; v_last = v;
    end
    if strcmp(method, 'adadelta')
        alpha = learning_rate;
        avg_gk = beta1*avg_gk + (1 - beta1)*(gk.^2);
        pk = -(sqrt(avg_pk + epsilon)./sqrt(avg_gk + epsilon)).*gk;
        avg_pk = beta1.*avg_pk + (1 - beta1).*(pk.^2);
        xk = xk + alpha*pk;
    end
    if strcmp(method,'lbfgs')
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
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%% Precond %%%%%%%%%%%%%%%%%%%%%%%%
        if args.Prec && k >= 1 && ( strcmp(method,'newton-qlp') || strcmp(method,'newton-cg') || strcmp(method,'gauss-newton') )
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
        
    if strcmp(method,'newton-qlp')
        assert( strcmp(args.subProbSolver, 'minres-qlp') || ...
            strcmp(args.subProbSolver, 'minres') || ...
            strcmp(args.subProbSolver, 'bicgstab') || ...
            strcmp(args.subProbSolver, 'symmlq') || ...
            strcmp(args.subProbSolver, 'sqmr') || ...
            strcmp(args.subProbSolver, 'cgs'));
        
        if strcmp(args.subProbSolver, 'minres-qlp')
            [pk,flagMINRESQLP,relresMINRESQLP,MINRESQLPiter] = myMINRESQLP(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'sqmr')
            [pk,flagMINRESQLP,relresMINRESQLP,MINRESQLPiter] = mySQMR(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'symmlq')
            [pk,flagMINRESQLP,relresMINRESQLP,MINRESQLPiter] = mySYMMLQ(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'minres')
            [pk,flagMINRESQLP,relresMINRESQLP,MINRESQLPiter] = minres(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'bicgstab')
            [pk,flagMINRESQLP,relresMINRESQLP,MINRESQLPiter] = bicgstab(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        if strcmp(args.subProbSolver, 'cgs')
            [pk,flagMINRESQLP,relresMINRESQLP,MINRESQLPiter] = cgs(Hk,-gk,subProbRelTol,subProbMaxItr,M);
        end
        %[pk,flagMINRESQLP,relresMINRESQLP,MINRESQLPiter] = myPCG(Hk,-gk,subProbRelTol,subProbMaxItr);
        current_props = current_props + MINRESQLPiter*2;
    end
    if strcmp(method,'newton-cg') || strcmp(method,'gauss-newton')
        if strcmp(method,'newton-cg')
            Ak = Hk;
        else
            assert(strcmp(method,'gauss-newton'));
            Ak = Gk;
        end
        if exact == true
            assert(false); % For now we are disabling the exact case
            assert(~isa(Ak,'function_handle'));
            pk = -Ak\gk;
        else
            [pk,flagPCG,relresPCG,PCGiter] = myPCG(Ak, -gk, subProbRelTol, subProbMaxItr);
            current_props = current_props + (1+PCGiter)*2;
            %assert(flagPCG == 0);
            if(isa(Ak,'function_handle'))
                assert(dot(pk,gk) <= -0.5*dot(pk,(Ak(pk))));
            else
                assert(dot(pk,gk) <= -0.5*dot(pk,(Ak*pk)));
            end
        end
    end
    if strcmp(method,'lbfgs')% then do line search
        alpha = args.alpha;
        beta1 = args.beta1;
        beta2 = args.beta2;
        [f_alpha,g_alpha] = objFun(xk+alpha*pk);
        itrLS = 0;
        while( ( ( f_alpha > ( fk + alpha*beta1*gk'*pk ) ) || abs(dot(g_alpha,pk)) > beta2*abs(dot(gk,pk)) )  && itrLS < linesearchMaxItrs )
            alpha = alpha/2;
            [f_alpha,g_alpha] = objFun(xk+alpha*pk);
            itrLS = itrLS + 1;
        end
        current_props = current_props + (itrLS+1)*2;
        xk = xk + alpha*pk;
        g_prev = gk;
        p_prev = pk;
        alpha_prev = alpha;
    end
    if strcmp(method,'newton-cg') ||  strcmp(method,'gauss-newton')% then do line search
        alpha = args.alpha;
        beta1 = args.beta1;
        f_alpha = objFun(xk+alpha*pk);
        itrLS = 0;
        while( ( f_alpha > ( fk + alpha*beta1*dot(gk,pk) ) ) && itrLS < linesearchMaxItrs )
            alpha = alpha/2;
            f_alpha = objFun(xk+alpha*pk);
            itrLS = itrLS + 1;
        end
        current_props = current_props + (itrLS+1);
        xk = xk + alpha*pk;
        g_prev = gk;
        p_prev = pk;
        alpha_prev = alpha;
    end
    if strcmp(method,'newton-qlp') % then do line search
        alpha = args.alpha;
        beta1 = args.beta1;
        beta2 = args.beta2;
        [~,g_alpha] = objFun(xk+alpha*pk);
        itrLS = 0;
        if args.Prec && k >= 1
            while( ( ( ( norm(g_alpha) )^2 )  >  ( ( ( norm(gk) )^2 ) + 2*alpha*beta1*dot(pk, Hk(gk)) )  || dot(g_alpha,pk) <= beta2*dot(gk,pk) )  && itrLS < linesearchMaxItrs )
                alpha = alpha/2;
                [~,g_alpha] = objFun(xk+alpha*pk);
                itrLS = itrLS + 1;
            end
        else
            while( ( ( ( norm(g_alpha) )^2 )  >  ( ( ( norm(gk) )^2 ) + 2*alpha*beta1*dot(pk, Hk(gk)) )  )  && itrLS < linesearchMaxItrs )
                alpha = alpha/2;
                [~,g_alpha] = objFun(xk+alpha*pk);
                itrLS = itrLS + 1;
            end
        end
        current_props = current_props + (itrLS+1)*2;
        xk = xk + alpha*pk;
        g_prev = gk;
        p_prev = pk;
        alpha_prev = alpha;
    end
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    hist.elapsed_time = [hist.elapsed_time, hist.elapsed_time(end) + current_elapsed_time];
    k = k + 1;
    hist.props = [hist.props, hist.props(end) + current_props];
end
x = xk;
end