function [f,g,Hv,H] = softMaxFunReg(w, X, Y, regFun, subSampArgs)
%%
% @X is the (n x d) data matrix.
% @w is the (d x C) by 1 weight vector where C is the number of classes (% Technically the total number of classes is C+1, but the degree of freedom is only C).
% @Y is the (n x C) label matrix, i.e., Y(i,b) is one if i-th label is class
%    b, and other wise 0.

%% Example to generate data to test:
% n=10; d = 5; 
% total_no_of_C = 3; 
% X = randn(n,d); 
% I = eye(total_no_of_C,total_no_of_C-1); 
% ind = randsample(total_no_of_C,n,true); Y = I(ind, :); 
% w = randn(d*(total_no_of_C-1),1); 
% derivativeTest(@(w) softMaxFun(X,Y,w),randn(d*(total_no_of_C-1),1))

if nargin==0
    tic;
    runTest
    toc;
    return;
end


%%
if nargin <= 4
    subSamp_grad = false;
    subSamp_Hess = false;
else
    subSamp_grad = subSampArgs.grad;
    subSamp_Hess = subSampArgs.Hess;
end
%%
%%%%%% Sanity Checks %%%%%%
assert(length(w) == length(shiftdim(w))); % to make sure w is a column vector
[n,d] = size(X);
assert(ceil(length(w)/d) == floor(length(w)/d)); % to make sure that w is of a vector of length (C x d) for some integer C, where C is the number of classes.
C = length(w)/d; % Technically the total number of classes is C+1, but the degree of freedom is only C
W = (vec2mat(w,d))'; % A (d x C) matrix formed from w where each column is a w_b for class b, b = 1,2,C
assert(size(W,1) == d);
assert(size(W,2) == C);
assert(size(Y,1) == n);
assert(size(Y,2) == C);
labels = sort( unique(sum(Y,2)) );
assert(length(labels) <= 2); % Y can only have zeros and ones
if length(labels) > 1
    assert( labels(1) == 0 );
    assert( labels(2) == 1);
else
    assert(labels == 0 || labels == 1);
end
assert(max(sum(Y,2)) >= 0 && max(sum(Y,2)) <=1); % each row of y has to have at most only one 1.
assert(sum(sum(Y,2) == ones(n,1)) >= 0 && sum(sum(Y,2) == ones(n,1))<=n); % each row of y has to have at most only one 1.


%%
%%%%% Function Value %%%%%
XW = X*W; % (n x C) matrix
% Do the Log-Sum-Trick...Our formulation is like "log ( 1 + sum_{c=1}^{C-1}
% exp<x_{i}, w_{c}> )"
large_vals = max(0,max(XW,[],2));
XW_trick = XW - repmat(large_vals,1,C);
XW_1_trick = [-large_vals, XW_trick]; %To account for the "1" in the sum, i.e., exp(0)
sum_exp_trick = sum(exp(XW_1_trick),2);
log_sum_exp_trick = large_vals + log (sum_exp_trick); % (n x 1)
f_fun = sum(log_sum_exp_trick) - sum(sum(XW.*Y,2));

f_reg = 0;
g_reg = 0;
Hv_reg = @(v) 0;
H_reg = 0;

if exist('regFun','var') && ~isempty(regFun)
    assert(isa(regFun,'function_handle'));
    if nargout == 4
        [f_reg,g_reg,Hv_reg,H_reg] = regFun(w);
    end
    if nargout == 3
        [f_reg,g_reg,Hv_reg] = regFun(w);
    end
    if nargout == 2
        [f_reg,g_reg] = regFun(w);
    end
    if nargout == 1
        f_reg = regFun(w);
    end
end
f = f_fun/n + f_reg;
%%
%%%%% Gradient %%%%%
if nargout >= 2
    if subSamp_grad
        assert(subSampArgs.grad == true && subSampArgs.gradProp > 0 && subSampArgs.gradProp < 1);
        %perform the sampling here.
        s_g = floor( subSampArgs.gradProp * n );
        idx = randsample(n,s_g);
        scale = ( 1 / s_g );
        X_sub = X(idx,:);
        inv_sum_exp = 1./sum_exp_trick( idx, : );
        inv_sum_exp = repmat(inv_sum_exp,1,C);
        S = inv_sum_exp.*exp(XW_trick( idx, : )); % (s_g x C) matrix
        assert(sum(sum((S <= 1),2)) == s_g*C); % every entry has to be smaller than 1!
        g_fun = (X_sub')*(S - Y(idx, :));  %(d x C) matrix
        %s_g = floor( subSampArgs.gradProp * size(X, 1) );
        %scale = ( 1 / s_g );
        %cols = randi( [1,size(X,1)], 1, s_g );
        %rows = 1:s_g;
        %X_sample = sparse( rows, cols, ones( 1, s_g ), s_g, n );
        %inv_sum_exp = 1./sum_exp_trick( cols, : );
        %inv_sum_exp = repmat(inv_sum_exp,1,C);
        %S = inv_sum_exp.*exp(XW_trick( cols, : )); % (n x C) matrix
        %assert(sum(sum((S <= 1),2)) == s_g*C); % every entry has to be smaller than 1!
        %g_fun = (X_sample * X)'*(S - Y(cols, :)); %(d x C) matrix
    else
        assert(subSamp_grad == false);
        scale = 1/n;
        inv_sum_exp = 1./sum_exp_trick;
        inv_sum_exp = repmat(inv_sum_exp,1,C);
        S = inv_sum_exp.*exp(XW_trick); % (n x C) matrix
        assert(sum(sum((S <= 1),2)) == n*C); % every entry has to be smaller than 1!
        g_fun = (X')*(S - Y); %(d x C) matrix
    end
    g = scale*g_fun(:) + g_reg;        
end

%%
%%%%% Hessian-Vector Multiply %%%%%
if nargout >= 3
    if subSamp_Hess
        assert(subSampArgs.Hess == true && subSampArgs.HessProp > 0 && subSampArgs.HessProp < 1);
        s_H = floor( subSampArgs.HessProp * n );
        idx = randsample(n,s_H);
        scale = ( 1 / s_H );
        X_sub = X(idx,:);  				% picking the sampled data points
        inv_sum_exp = 1./sum_exp_trick( idx, : );
        %inv_sum_exp = repmat(inv_sum_exp,1,size(XW_trick,2));
        S = inv_sum_exp.*exp(XW_trick( idx, : ));
        Hv_fun = @(v) Hessian_Vector_Product(X_sub,S,v);
    else
        assert(subSamp_Hess == false);
        Hv_fun = @(v) Hessian_Vector_Product(X,S,v);
        scale = 1/n;
    end
    Hv = @(v) scale*Hv_fun(v) + Hv_reg(v);
end

%%%%% Explicite Hessian %%%%%
if nargout == 4
    S_cell = mat2cell(S,n,ones(C,1));
    SX_cell = cellfun(@(ww) spdiags(ww,0,n,n)*X, S_cell,'UniformOutput', false);
    SX_self_cell = cellfun(@(ww) (X')*ww, SX_cell,'UniformOutput', false);
    SX_cross = cell2mat(SX_cell);
    SX_cross = (SX_cross')*SX_cross;
    SX_self = blkdiag(SX_self_cell{:});
    H_fun = SX_self - SX_cross; % ( (d x C) x (d x C) ) matrix where each block is (d x d)
    H = H_fun/n + H_reg;
end

end

function Hv = Hessian_Vector_Product(X, B, v)
[~,d] = size(X);
assert(ceil(length(v)/d) == floor(length(v)/d)); % to make sure that v is of a vector of length (C x d) for some integer C, where C is the number of classes.
C = length(v)/d; % Technically the total number of classes is C+1, but the degree of freedom is only C
V = (vec2mat(v,d))'; % a (d x C) matrix formed from w where each column is a w_b for class b, b = 1,2,C
assert(size(X,2) == size(V,1));
assert(size(B,2) == size(V,2));
A = X*V; % (n x c) matrix
AB = A.*B;
XVd1W = AB - B.*(repmat(sum(AB,2),1,C));
Hv = reshape( (X')*XVd1W , d*C, 1);
end

function runTest
%% Example to generate data to test:
n=10; d = 5; 
total_no_of_C = 3; 
X = randn(n,d); 
I = eye(total_no_of_C,total_no_of_C-1); 
ind = randsample(total_no_of_C,n,true); Y = I(ind, :); 
%[X, Y] = loadData('mnist'); [~,d] = size(X); total_no_of_C = size(Y,2)+1;
derivativeTest(@(w) softMaxFunReg(w,X,Y),randn(d*(total_no_of_C-1),1))
end
