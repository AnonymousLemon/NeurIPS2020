function [f,g,H,Gv] = nlsFunReg(w,X,y,phi,regularizer)
global sampleFraction;
global sampleSize;
global sampleScheme;
global sampleFractionDet;

if nargin==0
    runTest
    return
end
n = size(X,1);
p = length(w);
Xw = X*w;
if exist('phi','var') && ~isempty(regularizer)
    assert(isa(phi,'function_handle'));
    if nargout == 3
        [f_phi,g_phi,H_phi] = phi(Xw);
    end
    if nargout == 2 || nargout == 4
        [f_phi,g_phi] = phi(Xw);
    end
    if nargout == 1
        f_phi = phi(Xw);
    end
end
if exist('regularizer','var') && ~isempty(regularizer)
    assert(isa(regularizer,'function_handle'));
    if nargout >= 3
        [f_reg,g_reg,~,H_reg] = regularizer(w);
    end
    if nargout == 2
        [f_reg,g_reg] = regularizer(w);
    end
    if nargout == 1
        f_reg = regularizer(w);
    end
end

f = 0.5*sum((f_phi - y).^2)/n + f_reg;

if nargout >= 2
    g = X'*(g_phi.*(f_phi - y))/n + g_reg;
end

if nargout == 3
    %%
    
    
    
    if strcmp(sampleScheme,'LS') || strcmp(sampleScheme,'LS-Fast')  || strcmp(sampleScheme,'Uniform') || strcmp(sampleScheme,'LS-2') || strcmp(sampleScheme,'RN') || strcmp(sampleScheme,'LS-Det')
        D = g_phi.*g_phi + H_phi.*(f_phi - y);
        B = sqrt( D ).*X;
        
%         %%%%%%%%%%%%%%%%%%%%%%%%%%
%         [Q,~] = qr(X,0);
%         lev1 = sum(Q.*Q,2);
%         Q = X;
%         row1 = sum(Q.*Q,2);
%         
% %         [Q,~] = qr(D.*X,0);
% %         Q = D.*X;
% %         lev2 = sum(Q.*Q,2);
%         
%         [Q,~] = qr(sqrt(D).*X,0);
%         lev2 = sum(conj(Q).*Q,2);
%         
%         Q = sqrt(D).*X;
%         row2 = sum(conj(Q).*Q,2);
%         
%         figure(1); plot(lev1);
%         figure(2); plot(row1);
%         figure(3); plot(lev2);
%         figure(4); plot(row2)
% %         figure(2); plot(D);
% %         figure(3); hist(D)
%         autoArrangeFigures(2,2,1);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if ~isreal(B)
            warning('off','backtrace')
            warning('The matrix is complex');
%             D = sort(eig(B.'*B));
%             Dt = D;
%             Dt(Dt>0) = log10(1+Dt(Dt>0));
%             Dt(Dt<0) = -log10(1-Dt(Dt<0));
%             plot(Dt);
%             pause;
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
            prob = sum(conj(Q).*Q,2)/p;
            
            % %%%%%%%%%%%%
            %             prob = min(s*lev,1);
            %             samples = find((rand(n,1) < prob) == 1);
            %             sampleSize = length(samples);
            %             sB = sqrt((1./(n*prob(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
            samples = randsample(n,s,true,prob);
            sampleSize = length(samples);
            sB = sqrt((1./(sampleSize*n*prob(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
        end
        
        if strcmp(sampleScheme,'LS-Det')
                       
            [Q,~] = qr(B,0);
            prob = sum(conj(Q).*Q,2)/p;
            [~,I] = sort(prob,'descend');
            
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             I0 = I(1:10*p);
%             II0 = setdiff(1:n,I0);
%             F = B(I0,:); 
%             G = B(II0,:);
%             E = (F.')*(F);
%             N = (G')*G;
%             fprintf('N/E: %g\n',norm(N)/norm(E));
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            s = ceil(sampleFraction*n);
            sDet = ceil(sampleFractionDet*s);
            sRand = s - sDet;
            
            sB_Det = sqrt(1/n).*B(I(1:sDet),:);
            
            II = setdiff(1:n,I(1:sDet));
            nn = length(II);
            BB = B(II,:);
            [Q,~] = qr(BB,0);
            prob = sum(conj(Q).*Q,2)/p;
            samples = randsample(nn,sRand,true,prob);
            sampleSize = length(samples);
            sB_Rand = sqrt((1./(sampleSize*n*prob(samples)))).*BB(samples,:);
            
            sampleSize = s;
            sB = [sB_Det; sB_Rand];
            
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
            
            prob = sum(Omega.*Omega,2);
            prob = prob/sum(prob);
            % %%%%%%%%%%%%
            
            
            
            %%%%%%%%%%%%%
            samples = randsample(n,s,true,prob);
            sampleSize = length(samples);
            sB = sqrt((1./(sampleSize*n*prob(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
        end
        if strcmp(sampleScheme,'LS-2')
            assert(false);
            indPos = find(D >= 0);
            nPos = length(indPos);
            s = ceil(sampleFraction*nPos);
            B_pos = sqrt( D(indPos) ).*X(indPos,:);
            [Q,~] = qr(B_pos,0);
            prob = sum(conj(Q).*Q,2)/p;
            prob = min(s*prob,1);
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
            prob = sum(conj(Q).*Q,2)/p;
            prob = min(s*prob,1);
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
        D = g_phi.*g_phi + H_phi.*(f_phi - y);
        B = D.*X;
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
        %         sB = B/sqrt(n);
        % %         Hv =@(v)  X'*( ( (g_phi.*g_phi) + H_phi.*(f_phi - y) ).*(X*v) )/n + Hv_reg(v);
        % %         return;
        %         norm(X'* ( ( (g_phi.*g_phi) + H_phi.*(f_phi - y) ).*X )/n-(sB.')*sB)
        %%%%%%%%%%%%
        
        %%
        sampleSize = n;
        H = X'*( ( (g_phi.*g_phi) + H_phi.*(f_phi - y) ).*(X) )/n + H_reg;
        
        %%%%%%%%%%%%%% Test %%%%%%%%%%%%%%%%%%%%%%%
        D = g_phi.*g_phi + H_phi.*(f_phi - y);
        B = sqrt( D ).*X;
        if ~isreal(B)
            warning('off','backtrace')
            warning('The matrix is complex');
        end
        %%%%%%%%%%%%%% End Test %%%%%%%%%%%%%%%%%%%%%%%
    end
end

if nargout == 4
    H = []; % Just so Matlab does not complain!
    Gv =@(v)  X'*( ( (g_phi.*g_phi)  ).*(X*v) )/n + H_reg(v);
end

end

function C = srht (A , s )
n = size (A, 2 ) ;
sgn = randi ( 2 , [ 1 , n ] ) * 2 - 3 ; % one half are +1 and the rest are ?1
A = bsxfun ( @times , A, sgn ) ; % flip the signs of each column w. p . 50%
n = 2 ^ ( ceil ( log2 ( n ) ) ) ;
C = ( fwht (A' , n ) )' ; % fast Walsh?Hadarmard transform
idx = sort ( randsample ( n , s ) ) ;
C = C ( : , idx ) ; % sub sampling
C = C * ( n / sqrt ( s ) ) ;
end

function runTest
close all; clear all; clc; rehash;
global sampleFraction;
global sampleScheme;
sampleFraction = 1;
sampleScheme = 'Full';
n=1000; d = 500;
X = randn(n,d);
I = eye(2,2-1);
ind = randsample(2,n,true); y = I(ind, :);
lambda = 0;
regularizer = @(w) regularizerL2(w, lambda);
derivativeTest(@(w) nlsFunReg(w,X,y,@(w) sigmoid (w),regularizer),randn(d,1))
end