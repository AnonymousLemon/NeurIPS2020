function [f,g,H] = tukeyBiweightFunReg(w,X, y, regFun)
global sampleFraction;
global sampleSize;
global sampleScheme;


if nargin==0
    tic;
    runTest
    toc;
    return;
end
n = size(X,1);
f_reg = 0;
g_reg = 0;
H_reg = 0;
if exist('regFun','var') && ~isempty(regFun)
    assert(isa(regFun,'function_handle'));
    if nargout >= 3
        [f_reg,g_reg,~,H_reg] = regFun(w);
    end
    if nargout == 2
        [f_reg,g_reg] = regFun(w);
    end
    if nargout == 1
        f_reg = regFun(w);
    end
end
y = X*w-y;

f = sum( ( y.^2 ) ./ ( y.^2 + 1 ) )/n + f_reg;

if nargout >= 2
    g = (X')*( ( 2*y )./( (y.^2 + 1).^2 ) ) / n + g_reg;
end

if nargout == 3
    
    %%
    n = size(X,1);
    p = size(X,2);
    
    
    if strcmp(sampleScheme,'LS') || strcmp(sampleScheme,'LS-Fast')  || strcmp(sampleScheme,'Uniform') || strcmp(sampleScheme,'LS-2') || strcmp(sampleScheme,'RN')
        D = ( -(2*(3*y.^2 - 1)) ./ ( (y.^2 + 1).^3 )  );
        B = sqrt( D ).*X;
        
        if ~isreal(B)
            warning('off','backtrace')
            warning('The matrix is complex');
        end
        
        if strcmp(sampleScheme,'RN')
            s = ceil(sampleFraction*n);
            prob = sum(conj(B).*B,2);
            prob = prob/sum(prob);
            samples = randsample(n,s,true,prob);
            sampleSize = length(samples);
            sB = sqrt((1./(sampleSize*n*prob(samples)))).*B(samples,:);
            
        end
        
        if strcmp(sampleScheme,'LS')
            s = ceil(sampleFraction*n);
            [Q,~] = qr(B,0);
            lev = sum(conj(Q).*Q,2)/p;
            
            % %%%%%%%%%%%%
            %             prob = min(s*lev,1);
            %             samples = find((rand(n,1) < prob) == 1);
            %             sampleSize = length(samples);
            %             sB = sqrt((1./(n*prob(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
            samples = randsample(n,s,true,lev);
            sampleSize = length(samples);
            sB = sqrt((1./(sampleSize*n*lev(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
        end
        if strcmp(sampleScheme,'LS-Fast')
            s = ceil(sampleFraction*n);
            
            r1 = ceil(max(s,p*log(n)));
            
            %%%%%%%%%%%%%%%%%%%%%%
            %             tic;
            %             ss = randsample(n,r1,true);
            %             D = sign(randn(n,1));
            %             DX = D.*X;
            %             %DX = DX(ss,:);
            %             SX = fwht(DX)*(2^(ceil(log(n)/log(2))))/sqrt(n);
            %             SX1 = SX(ss,:);
            %             toc;
            %%%%%%%%%%%%%%%%%%%%%%
            
            SX = (srht(X' , r1 ))';
            [~,R] = qr(SX,0);
            
            r2 = ceil(log(n)/2);
            P2 = randn(p,r2)/sqrt(r2);
            Omega = X*(R\P2);
            
            lev = sum(Omega.*Omega,2);
            lev = lev/sum(lev);
            % %%%%%%%%%%%%
            
            
            
            %%%%%%%%%%%%%
            samples = randsample(n,s,true,lev);
            sampleSize = length(samples);
            sB = sqrt((1./(sampleSize*n*lev(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
        end
        if strcmp(sampleScheme,'LS-2')
            assert(false);
            indPos = find(D >= 0);
            nPos = length(indPos);
            s = ceil(sampleFraction*nPos);
            B_pos = sqrt( D(indPos) ).*X(indPos,:);
            [Q,~] = qr(B_pos,0);
            lev = sum(conj(Q).*Q,2)/p;
            prob = min(s*lev,1);
            samplesPos = find((rand(nPos,1) < prob) == 1);
            if isempty(samplesPos)
                sB_pos = [];
            else
                sB_pos = sqrt((1./(n*prob(samplesPos)))).*B_pos(samplesPos,:);
            end
            % %%%%%%%%%%%%
            
            indNeg = find(D < 0);
            nNeg = length(indNeg);
            s = ceil(sampleFraction*nNeg);
            B_Neg = sqrt( D(indNeg) ).*X(indNeg,:);
            [Q,~] = qr(B_Neg,0);
            lev = sum(conj(Q).*Q,2)/p;
            prob = min(s*lev,1);
            samplesNeg = find((rand(nNeg,1) < prob) == 1);
            if isempty(samplesNeg)
                sB_Neg = [];
            else
                sB_Neg = sqrt((1./(n*prob(samplesNeg)))).*B_Neg(samplesNeg,:);
            end
            sampleSize = length(samplesPos) + length(samplesNeg);
            sB = [sB_pos; sB_Neg];
        end
        if strcmp(sampleScheme,'Uniform')
            s = ceil(sampleFraction*n);
            samples = randsample(n,s,true);
            sampleSize = length(samples);
            sB = B(samples,:)/sqrt(s);
        end
        
        
        H = (sB.')*(sB) + H_reg;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         H0 = B.'*B/n;
%         Hfull = H0 + H_reg;
%         fprintf('Rank of X: %g, Rank of Full: %g, Rank of Sampled:%g, rel-norm of H: %g, Proj Grad: %g\n',rank(X), rank(Hfull),rank(H), norm(H0 - sB.'*sB)/norm(H0), norm((Hfull)*pinv(Hfull)*g)/norm(g));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    if strcmp(sampleScheme,'RN-MX') || strcmp(sampleScheme,'LS-MX')
        D = ( -(2*(3*y.^2 - 1)) ./ ( (y.^2 + 1).^3 )  );
        B = D .*X;
        if strcmp(sampleScheme,'RN-MX')
            s = ceil(sampleFraction*n);
            prob = sum(X.*X,2) + sum(B.*B,2);
            prob = prob/sum(prob);
            samples = randsample(n,s,true,prob);
            sampleSize = length(samples);
            sA = sqrt((1./(sampleSize*n*prob(samples)))).*X(samples,:);
            sB = sqrt((1./(sampleSize*n*prob(samples)))).*B(samples,:);
        end
        if strcmp(sampleScheme,'LS-MX')
            s = ceil(sampleFraction*n);
            [QB,~] = qr(B,0);
            [QX,~] = qr(X,0);
            prob = sum(QX.*QX,2) + sum(QB.*QB,2);
            prob = prob/sum(prob);
            samples = randsample(n,s,true,prob);
            sampleSize = length(samples);
            sA = sqrt((1./(sampleSize*n*prob(samples)))).*X(samples,:);
            sB = sqrt((1./(sampleSize*n*prob(samples)))).*B(samples,:);
        end
        H = (sA.')*(sB) + H_reg;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         H0 = X.'*B/n;
        %         Hfull = H0 + H_reg;
        %         fprintf('Rank of X: %g, Rank of Full: %g, Rank of Sampled:%g, rel-norm of H: %g, Proj Grad: %g\n',rank(X), rank(Hfull),rank(H), norm(H0 - sB.'*sB)/norm(H0), norm((Hfull)*pinv(Hfull)*g)/norm(g));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    if strcmp(sampleScheme,'Full')
        H = X'*( ( -(2*(3*y.^2 - 1)) ./ ( (y.^2 + 1).^3 )  ) .* X )/n + H_reg;
    end
    
end


% if nargout >= 3
%     H = @(v) A'*( ( -(2*(3*y.^2 - 1)) ./ ( (y.^2 + 1).^3 )  ) .* (A*v) )/n + Hv_reg(v);
% end



end


function runTest
%%%%%%%%%%%%%%%%%%% Derivative Tes t%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all; rehash;
global sampleFraction;
global sampleScheme;
sampleFraction = 1;
sampleScheme = 'Full';

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

