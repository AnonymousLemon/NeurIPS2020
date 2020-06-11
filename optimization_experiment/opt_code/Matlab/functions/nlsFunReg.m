function [f,g,H,Gv] = nlsFunReg(w,X,y,phi,regularizer)
global sampleFraction;
global sampleSize;
global sampleScheme;

if nargin==0
    runTest
    return
end
n = size(X,1);
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
    n = size(X,1);
    p = size(X,2);
    
    
    if strcmp(sampleScheme,'LS') || strcmp(sampleScheme,'Uniform') || strcmp(sampleScheme,'LS-2')
        D = g_phi.*g_phi + H_phi.*(f_phi - y);
        B = sqrt( D ).*X;
        
        if ~isreal(B)
            warning('off','backtrace')
            warning('The matrix is complex');
        end
        
        if strcmp(sampleScheme,'LS')
            s = ceil(sampleFraction*n);
            [Q,~] = qr(B,0);
            lev = sum(conj(Q).*Q,2)/p;
            % %%%%%%%%%%%%
            prob = min(s*lev,1);
            samples = find((rand(n,1) < prob) == 1);
            sampleSize = length(samples);
            sB = sqrt((1./(n*prob(samples)))).*B(samples,:);
            %%%%%%%%%%%%%
            % samples = randsample(n,s,true,lev);
            % sB = sqrt((1./(n*lev(samples)))).*B(samples,:);
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
        H0 = B.'*B/n;
        Hfull = H0 + H_reg;
        fprintf('Rank of Full: %g, Rank of Sampled:%g, rel-norm of H: %g, Proj Grad: %g\n',rank(Hfull),rank(H), norm(H0 - sB.'*sB)/norm(H0), norm((Hfull)*pinv(Hfull)*g)/norm(g));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    else
        assert(strcmp(sampleScheme,'Full'));
        %         sB = B/sqrt(n);
        % %         Hv =@(v)  X'*( ( (g_phi.*g_phi) + H_phi.*(f_phi - y) ).*(X*v) )/n + Hv_reg(v);
        % %         return;
        %         norm(X'* ( ( (g_phi.*g_phi) + H_phi.*(f_phi - y) ).*X )/n-(sB.')*sB)
        %%%%%%%%%%%%
        
        %%
        sampleSize = n;
        H = X'*( ( (g_phi.*g_phi) + H_phi.*(f_phi - y) ).*(X) )/n + H_reg;
    end
end

if nargout == 4
    H = []; % Just so Matlab does not complain!
    Gv =@(v)  X'*( ( (g_phi.*g_phi)  ).*(X*v) )/n + H_reg(v);
end

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