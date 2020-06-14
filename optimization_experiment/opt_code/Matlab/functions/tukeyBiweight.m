function [f,g,H] = tukeyBiweight(x,A, b, alpha, regFun)
if nargin==0
    tic;
    runTest
    toc;
    return;
end
n = size(A,1);
f_reg = 0;
g_reg = 0;
Hv_reg = @(v) 0;
if exist('regFun','var') && ~isempty(regFun)
    assert(isa(regFun,'function_handle'));
    if nargout >= 3
        [f_reg,g_reg,Hv_reg] = regFun(x);
    end
    if nargout == 2
        [f_reg,g_reg] = regFun(x);
    end
    if nargout == 1
        f_reg = regFun(x);
    end
end
d_high = 1;
d_low = -1; 
y = A*x-b;
idx3 = y > d_high;
idx2 = (y > d_low) & (y <= d_high);
idx1 = y <= d_low;
assert(sum(idx3 & idx2) == 0);
assert(sum(idx3 & idx1) == 0);
assert(sum(idx2 & idx1) == 0);

y1 = y(idx1);
y2 = y(idx2);
y3 = y(idx3);


f1 = 1/6*(-y1).^(alpha);%(idx1 +0);
f2 = (1/6)*(1-(1-y2.^2).^3);
f3 = 1/6*y3.^(alpha);%(idx3 +0);

f = ( sum(f1) + sum(f2) + sum(f3) )/n + f_reg ;
if nargout >= 2
    A1 = A(idx1,:);
    A2 = A(idx2,:);
    A3 = A(idx3,:);
    g1 = 0; g2 = 0; g3 = 0; 
    if ~isempty(A1)
        g1 = -alpha*A1'*((-y1).^(alpha-1))/6;
    end
    if ~isempty(A2)
        g2 = A2'*(y2.*(y2.^2 - 1).^2);
    end
    if ~isempty(A3)
        g3 = alpha*A3'*((y3).^(alpha-1))/6;
    end
    g = ( g1 + g2 + g3 ) / n + g_reg;
end

idx1 = idx1 +0; idx1(idx1 == 1) = alpha*(alpha-1)*((-y1).^(alpha-2))/6;
idx2 = idx2 +0;idx2(idx2 ==1) = 5*y2.^4 - 6*y2.^2 + 1;
idx3 = idx3 +0; idx3(idx3 ==1) = alpha*(alpha-1)*(y3.^(alpha-2))/6;

if nargout >= 3
    H = @(v) A'*( ( idx1 + idx2 + idx3 ) .* (A*v) )/n + Hv_reg(v);
end

end


function runTest
%%%%%%%%%%%%%%%%%%%% Derivative Tes t%%%%%%%%%%%%%%%%%%%%%
% clc; clear all; close all; rehash;
% p = 100;
% n = 10;
% A = randn(n,p);
% b = ones(n,1);
% alpha = 0;
% derivativeTest(@(x) tukeyBiweight(x,A,b,alpha),randn(p,1))
% autoArrangeFigures(2,2,1)
% return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all; rehash;
d_high = 1;
d_low = -1; 
step = 1E-3;
x1 = (-4:step:d_low-step/2)';
a = 1;
b = 0;
alpha = 0;
lambda = 0;
regFun = @(x) regularizerL2(x, lambda);
f1 = zeros(length(x1),1); g1 = zeros(length(x1),1); H1 = zeros(length(x1),1);
for i = 1:length(x1)
    [f1(i),g1(i),Hv1] = tukeyBiweight(x1(i),a,b,alpha,regFun);
    H1(i) = Hv1(1);
end

x2 = (d_low+step/2:step:d_high-step/2)';
f2 = zeros(length(x2),1); g2 = zeros(length(x2),1); H2 = zeros(length(x2),1);
for i = 1:length(x2)
    [f2(i),g2(i),Hv2] = tukeyBiweight(x2(i),a,b,alpha,regFun);
    H2(i) = Hv2(1);
end

x3 = (d_high+step/2:step:4)';
f3 = zeros(length(x3),1); g3 = zeros(length(x3),1); H3 = zeros(length(x3),1);
for i = 1:length(x3)
    [f3(i),g3(i),Hv3] = tukeyBiweight(x3(i),a,b,alpha,regFun);
    H3(i) = Hv3(1);
end

linewidth = 2;
fontSize = 12;

figure(1);
plot(x1,f1,'b', 'LineWidth', linewidth);  
set(gca,'fontsize',fontSize, 'fontweight','bold'); hold on; 
plot(x2,f2,'r','LineWidth', linewidth); plot([x1(end) x1(end)],[f2(1), f1(end)],'k--');
plot(x3,f3,'b','LineWidth', linewidth); plot([x2(end) x2(end)],[f2(end), f3(1)],'k--')
plot(x1(end), f1(end), 'b.', 'MarkerSize',27);
plot(x1(end), f2(1), 'ro', 'LineWidth', 1, 'MarkerSize',8);
plot(x2(end), f2(end), 'r.', 'MarkerSize',27);
plot(x2(end), f3(1), 'bo', 'LineWidth', 1, 'MarkerSize',8);
plot(0, tukeyBiweight(0,a,b,alpha,regFun), 'g.', 'MarkerSize',27);
xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$f(x)\quad \quad$$','interpreter','latex'); 

% g1 = abs(g1);
% g2 = abs(g2);
% g3 = abs(g3);

figure(2);
plot(x1,g1,'b','LineWidth', linewidth); 
set(gca,'fontsize',fontSize, 'fontweight','bold'); hold on; 
plot(x2,g2,'r','LineWidth', linewidth); plot([x1(end) x1(end)],[g2(1), g1(end)],'k--');
plot(x3,g3,'b','LineWidth', linewidth); plot([x2(end) x2(end)],[g2(end), g3(1)],'k--')
plot(x1(end), g1(end), 'b.', 'MarkerSize',27);
plot(x1(end), g2(1), 'ro', 'LineWidth', 1, 'MarkerSize',8);
plot(x2(end), g2(end), 'r.', 'MarkerSize',27);
plot(x2(end), g3(1), 'bo', 'LineWidth', 1, 'MarkerSize',8);
xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$\frac{d f(x)}{d x}$$','interpreter','latex'); 

figure(3);
plot(x1,H1,'b','LineWidth', linewidth); 
set(gca,'fontsize',fontSize, 'fontweight','bold'); hold on; %ylim([-1.1 1.1]);
plot(x2,H2,'r','LineWidth', linewidth); plot([x1(end) x1(end)],[H2(end), H1(1)],'k--')
plot(x3,H3,'b','LineWidth', linewidth); plot([x2(end) x2(end)],[H2(end), H3(1)],'k--')
plot(x1(end), H2(1), 'ro', 'LineWidth', 1, 'MarkerSize',8);
plot(x1(end), H1(end), 'b.', 'MarkerSize',27);
plot(x2(end), H2(end), 'r.', 'MarkerSize',27);
plot(x2(end), H3(1), 'bo', 'LineWidth', 1, 'MarkerSize',8);
xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$\frac{ d^2 f(x)}{d x^2}$$','interpreter','latex'); 

figure(4);
g1H1 = g1.*H1;
g2H2 = g2.*H2;
g3H3 = g3.*H3;
figure(4);
plot(x1,g1H1,'b','LineWidth', linewidth);  hold on;
set(gca,'fontsize',fontSize, 'fontweight','bold');
plot(x2,g2H2,'r','LineWidth', linewidth);  plot([x1(end) x1(end)],[g1H1(end) g2H2(1)],'k--')
plot(x3,g3H3,'b','LineWidth', linewidth); plot([x2(end) x2(end)],[g2H2(end), g3H3(1)],'k--')
plot(x1(end), g2H2(1), 'ro', 'LineWidth', 1, 'MarkerSize',8);
plot(x1(end), g1H1(end), 'b.', 'MarkerSize',27);
plot(x2(end), g2H2(end), 'r.', 'MarkerSize',27);
plot(x2(end), g3H3(1), 'bo', 'LineWidth', 1, 'MarkerSize',8);
xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$\left(\frac{d f(x)}{d x}\right) \times \left(\frac{d^2 f(x)}{d x^2}\right)$$','interpreter','latex'); 

dir_name = './results/tukeyBiweight';
if ~exist(dir_name, 'dir')
    mkdir(dir_name);
end
    
figure(1);  
saveas(gcf,[dir_name,'/','f'],'fig');
saveas(gcf,[dir_name,'/','f'],'png');
saveas(gcf,[dir_name,'/','f'],'pdf');

figure(2);  
saveas(gcf,[dir_name,'/','g'],'fig');
saveas(gcf,[dir_name,'/','g'],'png');
saveas(gcf,[dir_name,'/','g'],'pdf');

figure(3);  
saveas(gcf,[dir_name,'/','H'],'fig');
saveas(gcf,[dir_name,'/','H'],'png');
saveas(gcf,[dir_name,'/','H'],'pdf');

figure(4);  
saveas(gcf,[dir_name,'/','gH'],'fig');
saveas(gcf,[dir_name,'/','gH'],'png');
saveas(gcf,[dir_name,'/','gH'],'pdf');

autoArrangeFigures(2,2,1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

