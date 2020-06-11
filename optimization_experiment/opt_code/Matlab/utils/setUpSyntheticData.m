function dataStruct = setUpSyntheticData(problemType, syntheticProblemSize)
assert( any(strcmp(problemType,'gmm')) || ...
     any(strcmp(problemType,'invex01')) || ...
     any(strcmp(problemType,'invex02')) || ...
     any(strcmp(problemType,'invex03')) || ...
     any(strcmp(problemType,'invex04')) ||...
     any(strcmp(problemType,'tukeyBiweight')) || ...
     any(strcmp(problemType,'tukeyBiweightSmooth')) );

n = syntheticProblemSize(1);
p = syntheticProblemSize(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%% GMM %%%%%%%%%%%%%%%%%%%%%%%%%%
if any(strcmp(problemType,'gmm'))
    mu1 = -rand(p,1);
    mu2 = rand(p,1);
    alpha = 0.3;
    
    %C1 = 0.05*spdiags(logspace(0,2,p)',0,p,p)*sprandn(p,p,0.015);
    %C2 = 0.0135*spdiags(logspace(3,0,p)',0,p,p)*sprand(p,p,0.05);
    
    %C1 = spdiags(logspace(0,0,p)',0,p,p);
    %C2 = spdiags(logspace(0,0,p)',0,p,p);
    
    %C1 = spdiags(logspace(0,2,p)',0,p,p)*sprandn(p,p,0.1);
    %C2 = spdiags(logspace(3,0,p)',0,p,p)*sprand(p,p,0.1);
    
    A1 = randn(p,p); [Q1,~] = qr(A1); D1 = spdiags(logspace(0,1,p)',0,p,p); C1 = D1*Q1;
    A2 = rand(p,p);  [Q2,~] = qr(A2); D2 = spdiags(logspace(1,0,p)',0,p,p); C2 = D2*Q2;
    
    %C1 = randn(p,p);
    %C2 = rand(p,p);
    
    %C1 = speye(p,p);
    %C2 = speye(p,p);
    
    mu = [mu1';mu2'];
    Sigma = cat(3,inv(full(C1'*C1)),inv(full(C2'*C2)));
    gm = gmdistribution(mu,Sigma,[alpha, 1-alpha]);
    A_train = random(gm,n);
    d = 2*p+1;
    dataStruct.alpha =alpha;
    dataStruct.mu = mu;
    dataStruct.C1 = C1;
    dataStruct.C2 = C2;
    b_train = [];
    b_test = [];
    A_test = [];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% Invex %%%%%%%%%%%%%%%%%%%%%%%%%%
if ( any(strcmp(problemType,'invex01')) || ...
     any(strcmp(problemType,'invex02')) || ...
     any(strcmp(problemType,'invex03')) || ...
     any(strcmp(problemType,'invex04')) || ...
     any(strcmp(problemType,'tukeyBiweight')) ||...
     any(strcmp(problemType,'tukeyBiweightSmooth')) )
    A_train = randn(n,p);
    [U,~,V] = svd(A_train);
    D = spdiags(logspace(0,0,n)',0,n,p);
    A_train = U*D*V';
    b_train = A_train*rand(p,1) + 2*randn(n,1) + binornd(1,0.3,n,1);
    d = p;
    b_test = -2*ones(n,1);
    A_test = A_train;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataStruct.A_train = A_train;
dataStruct.b_train = b_train;
dataStruct.A_test = A_test;
dataStruct.b_test = b_test;
dataStruct.d = d;
dataStruct.p = p;
dataStruct.n_train = n;
end
