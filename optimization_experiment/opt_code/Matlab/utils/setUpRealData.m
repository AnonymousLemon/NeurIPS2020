function dataStruct = setUpRealData(problemType,dataName)

assert( any(strcmp(problemType,'nlls')) || ...
    any(strcmp(problemType,'softmax')) || ...
    any(strcmp(problemType,'tukeyBiweight')) || ...
    any(strcmp(problemType,'tukeyBiweightFunReg')) );

%%%%%%%%%%%%%%%%% Loading Data %%%%%%%%%%%%%%%%%%%%%%
[A_train, b_train, A_test, b_test] = loadData(dataName);
%A_train = normalizeData(A_train); A_test = normalizeData(A_test);
%A_train = standardizeData(A_train); A_test = standardizeData(A_test);

%A_train = A_train(1:1000,:);
%b_train = b_train(1:1000,:);

n = size(A_train,1);
p = size(A_train,2);
n_train = size(A_train,1);
n_test = size(A_test,1);
b_train = shiftdim(b_train);
assert(length(b_train) == n_train);

if any(strcmp(problemType,'nlls'))
    b_train = [b_train,1-sum(b_train,2)];
    b_train = b_train*([1:size(b_train,2)]');
    midian_label = median(unique(b_train));
    
    b_train(b_train <= midian_label) = 0;
    b_train(b_train > midian_label) = 1;
    
    b_test = b_test*([1:size(b_test,2)]');
    b_test(b_test <= midian_label) = 0;
    b_test(b_test > midian_label) = 1;
    b_test = [b_test 1-b_test];
end

if any(strcmp(problemType,'softmax')) || any(strcmp(problemType,'nlls'))
    nClasses = size(b_train,2)+1;
    d = p*(nClasses-1);
else
    assert(any(strcmp(problemType,'tukeyBiweight')) || ...
        any(strcmp(problemType,'tukeyBiweightFunReg')))
    nClasses = 0;
    d = p;
end
dataStruct.A_train = A_train;
dataStruct.b_train = b_train;
dataStruct.A_test = A_test;
dataStruct.b_test = b_test;
dataStruct.n_train = n_train;
dataStruct.n_test = n_test;
dataStruct.p = p;
dataStruct.nClasses = nClasses;
dataStruct.d = d;
end