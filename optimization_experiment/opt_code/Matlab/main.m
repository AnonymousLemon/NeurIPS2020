addpath(genpath(pwd));
close all; clear all; clc; rehash;
global sampleFraction;
global sampleScheme;


seed = 500; rng(seed);

sampleFractions = [0.05,0.05,1];
%method = 'trust-region';
method = 'newton-mr';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
problemType = {
    %'softmax'
    'nlls'
    %'tukeyBiweightFunReg'
    %'gmm'
    %'invex01'
    %'invex02'
    %'invex03'
    %'invex04'
    %'tukeyBiweight'
    };

dataNames = {
    %%%%%%%%%%%%%%%%%% Classification Data %%%%%%%%%%%%%%%%%%%%%%
    %     '20News';                       % Multi-Class:    n_train: 10142,   n_test: 1127,   p: 53975,   C: 20
    %     'arcene';                       % Binary:         n_train: 100,     n_test: 100,    p: 10000,   C: 2
    %     'cifar10';                      % Multi-Class:    n_train: 50000,   n_test: 10000,  p: 3072,    C: 10
    %     'covetype';                     % Multi-Class:    n_train: 435759,  n_test: 145253, p: 54,      C: 7
    %     'dorothea';                     % Binary:         n_train: 800,     n_test: 350,    p: 100000,  C: 2
         'drive-diagnostics';            % Multi-Class:    n_train: 50000,   n_test: 8509,   p: 48,      C: 11
    %     'gisette';                      % Binary:         n_train: 6000,    n_test: 1000,   p: 5000,    C: 2
    %     'hapt';                         % Multi-Class:    n_train: 7767,    n_test: 3162,   p: 561,     C: 12
    %     'mnist';                        % Multi-Class:    n_train: 60000,   n_test: 10000,  p: 784,     C: 10
    %     'UJIIndoorLoc-classification';  % Multi-Class:    n_train: 19937,   n_test: 1111,   p: 520,     C: 5
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% Regression Data %%%%%%%%%%%%%%%%%%%%%%
    %     'blogdata';                       % Regression:     n_train: 166, d: 50
    %     'power-plant';                    % Regression:     n_train: 8611, d: 4,
    %     'news-populairty';                % Regression:     n_train: 35679, d: 59
    %     'housing';                        % Regression:     n_train: 455, d: 13,
    %     'blog-feedback';                  % Regression:     n_train: 47157, d: 280
    %     'forest-fire';                    % Regression:     n_train: 465, d: 10,
    %     'UJIIndoorLoc-regression';        % Regression:     n_train: 19937, d: 520
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %    'synthetic'
    };

regParam = 1;

regType = {
    %'none'
    'convex'
    %'nonconvex'
    };

initialIterate = {
    %'randn'
    %'rand'
    %'ones'
    'zeros'
    };

%%%%%%%%%%%%%%%%%%% Optimization Hyper-parameters %%%%%%%%%%%%%%%%%%%%%%
%%% General parameters
args.maxItrs = 50;
args.gradTol = 1E-4;
args.maxProps = 1E8;
args.subProbMaxItr = 100;

%%% Line-search parameters
args.alpha = 1;
args.linesearchMaxItrs = 1E3;
args.subProbRelTol = 1E-6;%5E-2;
args.beta1 = 1E-4;
args.beta2 = 0.1;
args.Prec = false;

%%% Trust region parameters
args.delta = 0.1;
args.eta1 = 0.8;
args.eta2 = 1e-4;
args.gamma1 = 2;
args.gamma2 = 1.2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

legendText = {};
lineWidth = 2;
for i =1:length(sampleFractions)
    if i == 1
        sampleScheme = 'Uniform';
    end
    if i == 2
        sampleScheme = 'LS';
    end
    if i == 3
        sampleScheme = 'Full';
    end
    sampleFraction = sampleFractions(i);
    
    %% Steup the problem and data
    [objFun, x0, args] = initialize_and_setup(problemType,dataNames,regType,regParam,initialIterate,args);
    
    %% Optimization routine
    [~, hist]= minFunc(objFun, x0, args, method);
    
    %% Plotting
    legendText(end+1) = {sprintf('Newton-MR --- s/n: %g, Sampling: %s',sampleFraction,sampleScheme)};
    figure(1);
    loglog(hist.props,hist.objVal,'LineWidth', lineWidth);
    legend(legendText); hold on;
    xlabel('Oracle Calls'); ylabel('\textbf{$$F(\textbf{x})$$: Objective Function}','interpreter','latex');
    figure(2);
    loglog(hist.props,hist.gradNorm,'LineWidth', lineWidth);
    legend(legendText); hold on;
    xlabel('Oracle Calls'); ylabel('\textbf{$$\|\nabla F(\textbf{x})\|$$: Gradient Norm}','interpreter','latex');
    figure(3);
    loglog(hist.props,hist.testVal,'LineWidth', lineWidth);
    legend(legendText); hold on;
    xlabel('Oracle Calls'); ylabel('Test Classification Accuracy');
    
    autoArrangeFigures(2,2,1);
end