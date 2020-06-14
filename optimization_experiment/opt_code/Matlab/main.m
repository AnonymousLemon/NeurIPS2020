addpath(genpath(pwd));
close all; clear all; clc; rehash;
global sampleFraction;
global sampleFractionDet;
global sampleScheme;

s = 0.01;
sampleFractions = [s,s,s,s,s,1];
method = 'newton-cg';
problemType = {'nlls'};
dataName = {'covetype'};
regParam = 1;
regType = {'convex'};
initialIterate = {'zeros'};

%%%%%%%%%%%%%%%%%%% Optimization Hyper-parameters %%%%%%%%%%%%%%%%%%%%%%
%%% General parameters
args.maxItrs = inf;
args.gradTol = 2E-8;
args.maxProps = 1E7;
args.subProbMaxItr = 100;

%%% Line-search parameters
args.alpha = 1;
args.linesearchMaxItrs = 1E3;
args.subProbRelTol = 5E-6;%5E-2;
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
        sampleScheme = 'RN';
    end
    if i == 4
        sampleScheme = 'LS-MX';
    end
    if i == 5
        sampleScheme = 'RN-MX';
    end
    if i == 6
        sampleScheme = 'Full';
    end
    
    sampleFraction = sampleFractions(i);
    
    %% Steup the problem and data
    [objFun, x0, args, dir_name] = initialize_and_setup(problemType,dataName,regType,regParam,initialIterate,args);
    dir_name = [dir_name,'/',method];
    if ~exist(dir_name, 'dir')
        mkdir(dir_name);
    end
    
    %% Optimization routine
    [~, hist]= minFunc(objFun, x0, args, method);
    
    %% Plotting
    legendText(end+1) = {sprintf('%s --- s/n: %g, Sampling: %s',method,sampleFraction,sampleScheme)};
    figure(1);
    loglog(hist.props,hist.objVal,'LineWidth', lineWidth);
    legend(legendText,'Location','Best'); hold on;
    xlabel('Oracle Calls'); ylabel('\textbf{$$F(\textbf{x})$$: Objective Function}','interpreter','latex');
    figure(2);
    loglog(hist.props,hist.gradNorm,'LineWidth', lineWidth);
    legend(legendText,'Location','Best'); hold on;
    xlabel('Oracle Calls'); ylabel('\textbf{$$\|\nabla F(\textbf{x})\|$$: Gradient Norm}','interpreter','latex');
    figure(3);
    loglog(hist.props,hist.testVal,'LineWidth', lineWidth);
    legend(legendText,'Location','Best'); hold on;
    xlabel('Oracle Calls'); ylabel('Test Classification Accuracy');
    
    autoArrangeFigures(2,2,1);
    
    %%
    figure(1); axis tight; legend('Location','SouthWest');
    saveas(gcf,[dir_name,'/','Obj_Props'],'fig');
    saveas(gcf,[dir_name,'/','Obj_Props'],'png');
    saveas(gcf,[dir_name,'/','Obj_Props'],'pdf');
    
    figure(2); axis tight;
    saveas(gcf,[dir_name,'/','Grad_Props'],'fig');
    saveas(gcf,[dir_name,'/','Grad_Props'],'png');
    saveas(gcf,[dir_name,'/','Grad_Props'],'pdf');
    
    figure(3); axis tight;
    saveas(gcf,[dir_name,'/','Test_Props'],'fig');
    saveas(gcf,[dir_name,'/','Test_Props'],'png');
    saveas(gcf,[dir_name,'/','Test_Props'],'pdf');
    
    save([dir_name,'\','hist','_',sampleScheme],'hist')
    
end