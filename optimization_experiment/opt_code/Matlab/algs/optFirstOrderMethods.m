function [x,hist] = optFirstOrderMethods(objFun,x0,args, method)

assert( strcmp(method, 'gd') || ...
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
hist.gradNorm = [];
hist.testVal = zeros(1,1);
hist.elapsed_time = [];
xk = x0;
k = 0;

%% Get the arguments specific to each method
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
    current_props = current_props + 2;
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
    if norm(gk) < gradTol || k >= maxItrs || hist.props(end) > maxProps
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
    tt = toc; current_elapsed_time = current_elapsed_time + tt;
    hist.elapsed_time = [hist.elapsed_time, hist.elapsed_time(end) + current_elapsed_time];
    k = k + 1;
    hist.props = [hist.props, hist.props(end) + current_props];
end
x = xk;
end