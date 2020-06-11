function [f,g,Hv,H] = softMaxFunRegNonUniform(w, X, Y, regularizer)
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
large_vals = max(0,max(XW,[],2)); % ONLY WE CHECK FOR MAXIMUM POSTIVE VALUES!
XW_trick = XW - repmat(large_vals,1,C);
XW_1_trick = [-large_vals, XW_trick]; %To account for the "1" in the sum, i.e., exp(0)
sum_exp_trick = sum(exp(XW_1_trick),2);
log_sum_exp_trick = large_vals + log (sum_exp_trick); % (n x 1)
f_fun = sum(log_sum_exp_trick) - sum(sum(XW.*Y,2));

f_reg = 0;
g_reg = 0;
Hv_reg = @(v) 0;
H_reg = 0;

if exist('regularizer','var') && ~isempty(regularizer)
    assert(isa(regularizer,'function_handle'));
    if nargout == 4
        [f_reg,g_reg,Hv_reg,H_reg] = regularizer(w);
    end
    if nargout == 3
        [f_reg,g_reg,Hv_reg] = regularizer(w);
    end
    if nargout == 2
        [f_reg,g_reg] = regularizer(w);
    end
    if nargout == 1
        f_reg = regularizer(w);
    end
end
f = f_fun + f_reg;


%%
%%%%% Gradient %%%%%
%% 20% gradient sample size is fixed %%%
%%%

%{
if nargout >= 2
    inv_sum_exp = 1./sum_exp_trick;
    inv_sum_exp = repmat(inv_sum_exp,1,C);
    S = inv_sum_exp.*exp(XW_trick); % (n x C) matrix
    assert(sum(sum((S <= 1),2)) == n*C); % every entry has to be smaller than 1!
    g_fun = (X')*(S - Y); %(d x C) matrix
    g_fun = g_fun(:); % gradient vector
    g = g_fun + g_reg;
end
%}


if nargout >= 2
	 %perform the sampling here. 
	 sample_size = floor( 0.2 * size(X, 1) ); 
	% disp(sample_size); 

	 cols = randi( [1,size(X,1)], 1, sample_size ); 
	 rows = 1:sample_size; 
	 X_sample = sparse( rows, cols, ones( 1, sample_size ), sample_size, n ); 

    inv_sum_exp = 1./sum_exp_trick( cols, : );
    inv_sum_exp = repmat(inv_sum_exp,1,C);
    S = inv_sum_exp.*exp(XW_trick( cols, : )); % (n x C) matrix
    assert(sum(sum((S <= 1),2)) == sample_size*C); % every entry has to be smaller than 1!
    %assert(sum(sum((S <= 1),2)) == n*C); % every entry has to be smaller than 1!

    g_fun = (X_sample * X)'*(S - Y(cols, :)); %(d x C) matrix
    g_fun = g_fun(:); % gradient vector
	 g_fun = ( n / sample_size ) * g_fun; 
    g = g_fun + g_reg;
end


%%
%%%%% Hessian-Vector Multiply %%%%%
if nargout >= 3
	 s = floor( 0.05 * n ); 
	 %disp(s); 

    inv_sum_exp = 1./sum_exp_trick;
	 S = inv_sum_exp.*exp(XW_trick);

	 D = getSecondDerivatives(S);% get the diagonal matrix of second derivatives, i.e., f''
	 D = sum( D, 2 ); 
% 	 rnorms = sqrt(sum(X.^2,2));
     rnorms = sum(X.^2,2);
	 p = abs(D).*rnorms;
	 p = full(p/sum(p));  					% compute the probability vector
	 q = min(1,p*s); 					% s is the number of samples
	 idx = find(rand(n,1)<q); 				% index of the selected ones
     p_sub = q(idx); 					% for re-weighting the hessian and the gradient
     scale = 1./(p_sub);
	 %idx = randsample(n,s,true,p); p_sub = p(idx); scale = 1./(s*p_sub);
     X_sub = X(idx,:);  				% picking the sampled data points


    %disp( min(p_sub) ); 

    inv_sum_exp = 1./sum_exp_trick( idx, : );
	 S = inv_sum_exp.*exp(XW_trick( idx, : ));

    Hv_fun = @(v) Hessian_Vector_Product(X_sub,S,v, scale);
    Hv = @(v) Hv_fun(v) + Hv_reg(v);
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
    H = H_fun + H_reg;
end

end

function Hv = Hessian_Vector_Product(X, B, v, scale)
[~,d] = size(X);
assert(ceil(length(v)/d) == floor(length(v)/d)); % to make sure that v is of a vector of length (C x d) for some integer C, where C is the number of classes.
C = length(v)/d; % Technically the total number of classes is C+1, but the degree of freedom is only C
V = (vec2mat(v,d))'; % a (d x C) matrix formed from w where each column is a w_b for class b, b = 1,2,C
assert(size(X,2) == size(V,1));
assert(size(B,2) == size(V,2));

X = X .* repmat( sqrt(scale), 1, size(X, 2));

A = X*V; % (n x c) matrix

%A = A .* repmat( scale, 1, size(A, 2));
%B = B .* repmat( 1 ./ scale, 1, size(B, 2));

AB = A.*B;
XVd1W = AB - B.*(repmat(sum(AB,2),1,C));
Hv = reshape( (X')*XVd1W , d*C, 1);
end


function Hdd = getSecondDerivatives( B )
	Hdd = B .* (1 - B);
end


