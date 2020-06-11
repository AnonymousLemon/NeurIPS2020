function [objFun, x0, args,dir_name] = initialize_and_setup(problemType,dataNames,regType,regParam,initialIterate,args)


assert(size(problemType,1) == 1, 'More than one problem type are selected.'); assert(size(problemType,2) == 1);
assert(size(regType,1) == 1, 'More than one regularization type are selected.'); assert(size(regType,2) == 1);
assert(size(initialIterate,1) == 1, 'None or More than one initial_iterate is selected.'); assert(size(initialIterate,2) == 1);
if any(strcmp(dataNames, 'synthetic'))
    assert(size(dataNames,1) == 1, 'Either selec synthetic or other data types.'); assert(size(dataNames,2) == 1);
end
assert(~ ( ( any(strcmp(problemType,'softmax')) || ...
    any(strcmp(problemType,'invex01')) || ...
    any(strcmp(problemType,'invex02')) || ...
    any(strcmp(problemType,'invex03')) || ...
    any(strcmp(problemType,'invex04')) || ...
    any(strcmp(problemType,'tukeyBiweight')) || ...
    any(strcmp(problemType,'tukeytukeyBiweightSmooth')) )), ...
    'Gauss-Newton does not apply to SOFTMAX, Invex or Tukey problem types.');

if ( any(strcmp(problemType,'gmm')) || ...
        any(strcmp(problemType,'invex01')) || ...
        any(strcmp(problemType,'invex02')) || ...
        any(strcmp(problemType,'invex03')) || ...
        any(strcmp(problemType,'invex04')) )
    assert(any(strcmp(dataNames, 'synthetic')), 'With GMM and Invex only synthetic data is supported.');
end
if any(strcmp(problemType,'softmax')) || any(strcmp(problemType,'nlls'))
    assert(~any(strcmp(dataNames, 'synthetic')), 'With SOFTMAX and NLLS synthetic data is not supported.');
end
if (any(strcmp(dataNames,'arcene')) || ... % classification problems
        any(strcmp(dataNames,'dorothea')) || ...
        any(strcmp(dataNames,'20News')) || ...
        any(strcmp(dataNames,'covetype')) || ...
        any(strcmp(dataNames, 'mnist')) || ...
        any(strcmp(dataNames, 'UJIIndoorLoc-classification')) || ...
        any(strcmp(dataNames, 'gisette')) || ...
        any(strcmp(dataNames, 'hapt')) || ...
        any(strcmp(dataNames, 'cifar10')) || ...
        any(strcmp(dataNames, 'drive-diagnostics')) )
    assert(strcmp(problemType{1},'softmax') || strcmp(problemType{1},'nlls'));
end
if (any(strcmp(dataNames, 'blogdata')) ||... % regression problems
        any(strcmp(dataNames, 'power-plant')) || ...
        any(strcmp(dataNames, 'news-populairty')) || ...
        any(strcmp(dataNames, 'housing')) || ...
        any(strcmp(dataNames, 'blog-feedback')) || ...
        any(strcmp(dataNames, 'forest-fire')) || ...
        any(strcmp(dataNames, 'UJIIndoorLoc-regression')))
    assert(strcmp(problemType{1},'tukeyBiweight') || strcmp(problemType{1},'tukeyBiweightFunReg'));
end

if any(strcmp(problemType,'gmm'))
    assert(~any(strcmp(regType,'nonconvex')));
end
%%
for i = 1:length(dataNames)
    %close all;
    dataName = dataNames{i};
    
    %%
    if any(strcmp(regType,'convex')) || regParam == 0
        if any(strcmp(problemType,'gmm'))
            regFun = @(x,lambda) regularizerL2(x,lambda,spdiags([sqrt(length(x(2:end))); ones(length(x)-1,1)],0,length(x),length(x)));
        else
            regFun = @(x,lambda) regularizerL2(x, lambda);
        end
    else
        regFun = @(x,lambda) regularizerNonConvex(x, lambda);
    end
    
    %%
    if strcmp(dataName, 'synthetic')
        p = 100;
        n = 1000;
        [objFun, testFun, dataStruct] = setUpProblem(problemType,dataName,regFun,regParam,[n,p]);
        if any(strcmp(problemType,'gmm'))
            assert(dataStruct.d == 2*p + 1);
        end
    else
        [objFun, testFun, dataStruct] = setUpProblem(problemType,dataName,regFun,regParam);
    end
    %%
    args.testFun = testFun;
    
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if any(strcmp(initialIterate,'randn'))
        x0 = randn(dataStruct.d,1);
    end
    if any(strcmp(initialIterate,'rand'))
        x0 = rand(dataStruct.d,1);
    end
    if any(strcmp(initialIterate,'ones'))
        x0 = ones(dataStruct.d,1);
    end
    if any(strcmp(initialIterate,'zeros'))
        x0 = zeros(dataStruct.d,1);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%
    if regParam == 0
        regType = {'none'};
    end
    saveName = [dataName,'_','n_',num2str(dataStruct.n_train),'_','d_',num2str(dataStruct.d),'_','Precond_',num2str(args.Prec),'_','Lambda_',num2str(regParam),'_','subMaxItr_',num2str(args.subProbMaxItr),'_','subRelTol_',num2str(args.subProbRelTol,'%1.0e'),'_','x0_',initialIterate{1}];
    dir_name = ['../results/',problemType{1},'_',regType{1},'/',saveName];
    if ~exist(dir_name, 'dir')
        mkdir(dir_name);
    end
    %file_name = [dir_name,'/',dataName];
    
    %%
    if any(strcmp(problemType,'nlls')) || any(strcmp(problemType,'softmax'))
        fprintf('\nProblem: %s, Reg Type: %s, Data: %s, n_train: %g, n_test: %g, p: %g, C: %g, Reg Param: %g, x0: %s\n', problemType{1}, regType{1}, dataName, dataStruct.n_train , dataStruct.n_test, dataStruct.p , dataStruct.nClasses, regParam, initialIterate{1});
    else
        %if any(strcmp(problemType,'gmm')) || any(strcmp(problemType,'invex'))
        fprintf('\nProblem: %s, Reg Type: %s, Data: %s, n_train: %g, d: %g, Reg Param: %g, x0: %s\n', problemType{1}, regType{1}, dataName, dataStruct.n_train , dataStruct.d , regParam, initialIterate{1});
    end

end
