function [objFun, testFun, dataStruct] = setUpProblem(problemType,dataName, regularizer, lambda, syntheticProblemSize)

assert( any(strcmp(problemType,'nlls')) || ...
    any(strcmp(problemType,'softmax')) || ...
    any(strcmp(problemType,'gmm')) || ...
    any(strcmp(problemType,'invex01')) || ...
    any(strcmp(problemType,'invex02')) || ...
    any(strcmp(problemType,'invex03')) || ...
    any(strcmp(problemType,'invex04')) ||...
    any(strcmp(problemType,'tukeyBiweight')) || ...
    any(strcmp(problemType,'tukeyBiweightFunReg')) );

if ~strcmp(dataName, 'synthetic')
    assert(nargin == 4);
    dataStruct = setUpRealData(problemType, dataName);
else
    assert(nargin == 5);
    dataStruct = setUpSyntheticData(problemType,syntheticProblemSize);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% Regularizer %%%%%%%%%%%%%%%%%%%%%%%%%%
regFun = @(x) regularizer(x,lambda/dataStruct.n_train);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%% SOFTMAX %%%%%%%%%%%%%%%%%%%%%%%%%%
if any(strcmp(problemType,'softmax'))
    objFun = @(x) softMaxFunReg(x, dataStruct.A_train, dataStruct.b_train, regFun);
    testFun = @(x) testSoftmaxClassification(dataStruct.A_test, dataStruct.b_test, x);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% NLLS %%%%%%%%%%%%%%%%%%%%%%%%%%
if any(strcmp(problemType,'nlls'))
    %phi = @(x) sigmoid(x);
    phi = @(x) tanhFun(x,1);
    objFun = @(x) nlsFunReg(x, dataStruct.A_train, dataStruct.b_train, phi,regFun);
    testFun = @(x) testNLLSClassification(dataStruct.A_test, dataStruct.b_test, phi, x);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% GMM %%%%%%%%%%%%%%%%%%%%%%%%%%
if any(strcmp(problemType,'gmm'))
    phi = @(x) tanhFun(x,1);
    %phi = @(x) sigmoid(x);
    objFun = @(x) gmmFunReg(x, dataStruct.A_train, dataStruct.C1, dataStruct.C2, phi,regFun);
    testFun = @(x) testGMMEstimation(x,phi,dataStruct.mu,dataStruct.alpha);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%% Tukey Biweight %%%%%%%%%%%%%%%%%%%%%%
if any(strcmp(problemType,'tukeyBiweight'))
    alpha = 0;
    objFun = @(x) tukeyBiweight(x,dataStruct.A_train, dataStruct.b_train, alpha, regFun);
    testFun = @(x) norm( dataStruct.A_test*x - dataStruct.b_test)/norm(dataStruct.b_test);
end

%%%%%%%%%%%%%%%%%%%%%%%% Tukey Biweight Smoothed %%%%%%%%%%%%
if any(strcmp(problemType,'tukeyBiweightFunReg'))
    objFun = @(x) tukeyBiweightFunReg(x,dataStruct.A_train, dataStruct.b_train, regFun);
    testFun = @(x) norm( dataStruct.A_test*x - dataStruct.b_test)/norm(dataStruct.b_test);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



end