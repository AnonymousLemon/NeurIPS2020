function [f,g,H] = tukeyBiweightFunReg(x,A, b, regFun)
global sampleSize;
global sampling;

if nargin==0
    tic;
    runTest
    toc;
    return;
end
n = size(A,1);
f_reg = 0;
g_reg = 0;
H_reg = 0;
if exist('regFun','var') && ~isempty(regFun)
    assert(isa(regFun,'function_handle'));
    if nargout >= 3
        [f_reg,g_reg,~,H_reg] = regFun(x);
    end
    if nargout == 2
        [f_reg,g_reg] = regFun(x);
    end
    if nargout == 1
        f_reg = regFun(x);
    end
end
y = A*x-b;

f = sum( ( y.^2 ) ./ ( y.^2 + 1 ) )/n + f_reg;

if nargout >= 2
    g = (A')*( ( 2*y )./( (y.^2 + 1).^2 ) ) / n + g_reg;
end

if nargout == 3
    
    %%
    n = size(A,1);
    p = size(A,2);
    
    
    if strcmp(sampling,'LS') || strcmp(sampling,'Uniform') || strcmp(sampling,'LS-2')
        D = ( -(2*(3*y.^2 - 1)) ./ ( (y.^2 + 1).^3 )  );
        B = sqrt( D ).*A;
        
        if ~isreal(B)
            warning('off','backtrace')
            warning('The matrix is complex');
        end
        
        if strcmp(sampling,'LS')
            s = ceil(sampleSize*n);
            [Q,~] = qr(B,0);
            lev = sum(conj(Q).*Q,2)/p;
            % %%%%%%%%%%%%
            prob = min(s*lev,1);
            samples = find((rand(n,1) < prob) == 1);
            sB = sqrt((1./(n*prob(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
            % samples = randsample(n,s,true,lev);
            % sB = sqrt((1./(n*lev(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
        end
        if strcmp(sampling,'LS-2')
            assert(false);
            indPos = find(D >= 0);
            nPos = length(indPos);
            s = ceil(sampleSize*nPos);
            B_pos = sqrt( D(indPos) ).*X(indPos,:);
            [Q,~] = qr(B_pos,0);
            lev = sum(conj(Q).*Q,2)/p;
            prob = min(s*lev,1);
            samples = find((rand(nPos,1) < prob) == 1);
            if isempty(samples)
                sB_pos = [];
            else
                sB_pos = sqrt((1./(n*prob(samples)))).*B_pos(samples,:);
            end
            % %%%%%%%%%%%%
            
            indNeg = find(D < 0);
            nNeg = length(indNeg);
            s = ceil(sampleSize*nNeg);
            B_Neg = sqrt( D(indNeg) ).*X(indNeg,:);
            [Q,~] = qr(B_Neg,0);
            lev = sum(conj(Q).*Q,2)/p;
            prob = min(s*lev,1);
            samples = find((rand(nNeg,1) < prob) == 1);
            if isempty(samples)
                sB_Neg = [];
            else
                sB_Neg = sqrt((1./(n*prob(samples)))).*B_Neg(samples,:);
            end
            sB = [sB_pos; sB_Neg];
        end
        if strcmp(sampling,'Uniform')
            s = ceil(sampleSize*n);
            samples = randsample(n,s,true);
            sB = B(samples,:)/sqrt(s);
        end
        
        
        H = (sB.')*(sB) + H_reg;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        H0 = B.'*B/n;
        Hfull = H0 + H_reg;
        fprintf('Rank of Full: %g, Rank of Sampled:%g, rel-norm of H: %g, Proj Grad: %g\n',rank(Hfull),rank(H), norm(H0 - sB.'*sB)/norm(H0), norm((Hfull)*pinv(Hfull)*g)/norm(g));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        else
        assert(strcmp(sampling,'Full'));
        H = A'*( ( -(2*(3*y.^2 - 1)) ./ ( (y.^2 + 1).^3 )  ) .* A )/n + H_reg;
    end
    
end


% if nargout >= 3
%     H = @(v) A'*( ( -(2*(3*y.^2 - 1)) ./ ( (y.^2 + 1).^3 )  ) .* (A*v) )/n + Hv_reg(v);
% end



end


function runTest
%%%%%%%%%%%%%%%%%%% Derivative Tes t%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all; rehash;
p = 100;
n = 10;
A = randn(n,p);
b = ones(n,1);
derivativeTest(@(x) tukeyBiweightFunReg(x,A,b),randn(p,1))
autoArrangeFigures(2,2,1)
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%% Plotting %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc; clear all; close all; rehash;
% a = 1;
% b = 0;
% x = (-5:0.001:5)';
% f = zeros(length(x),1); g = zeros(length(x),1); H = zeros(length(x),1);
% lambda = 0;
% regFun = @(x) regularizerL2(x, lambda);
% for i = 1:length(x)
%     [f(i),g(i),Hv] = tukeyBiweightSmooth(x(i),a,b,regFun);
%     H(i) = Hv(1);
% end
% 
% linewidth = 2;
% fontSize = 12;
% 
% figure(1);
% plot(x,f,'b', 'LineWidth', linewidth);  
% set(gca,'fontsize',fontSize, 'fontweight','bold'); hold on; 
% plot(0, tukeyBiweightSmooth(0,a,b), 'g.', 'MarkerSize',27);
% xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$f(x)\quad \quad$$','interpreter','latex'); 
% 
% figure(2);
% plot(x,g,'b','LineWidth', linewidth); 
% set(gca,'fontsize',fontSize, 'fontweight','bold'); hold on; 
% xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$\frac{d f(x)}{d x}$$','interpreter','latex'); 
% 
% figure(3);
% plot(x,H,'b','LineWidth', linewidth); 
% set(gca,'fontsize',fontSize, 'fontweight','bold'); hold on; 
% xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$\frac{ d^2 f(x)}{d x^2}$$','interpreter','latex'); 
% 
% figure(4);
% gH = g.*H;
% figure(4);
% plot(x,gH,'b','LineWidth', linewidth);  hold on;
% set(gca,'fontsize',fontSize, 'fontweight','bold');
% xlabel('\boldmath $$x$$','interpreter','latex'); title('\boldmath $$\left(\frac{d f(x)}{d x}\right) \times \left(\frac{d^2 f(x)}{d x^2}\right)$$','interpreter','latex'); 
% 
% dir_name = './results/tukeyBiweightSmooth';
% if ~exist(dir_name, 'dir')
%     mkdir(dir_name);
% end
%     
% figure(1);  
% saveas(gcf,[dir_name,'/','f'],'fig');
% saveas(gcf,[dir_name,'/','f'],'png');
% saveas(gcf,[dir_name,'/','f'],'pdf');
% 
% figure(2);  
% saveas(gcf,[dir_name,'/','g'],'fig');
% saveas(gcf,[dir_name,'/','g'],'png');
% saveas(gcf,[dir_name,'/','g'],'pdf');
% 
% figure(3);  
% saveas(gcf,[dir_name,'/','H'],'fig');
% saveas(gcf,[dir_name,'/','H'],'png');
% saveas(gcf,[dir_name,'/','H'],'pdf');
% 
% figure(4);  
% saveas(gcf,[dir_name,'/','gH'],'fig');
% saveas(gcf,[dir_name,'/','gH'],'png');
% saveas(gcf,[dir_name,'/','gH'],'pdf');
% 
% autoArrangeFigures(2,2,1)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

