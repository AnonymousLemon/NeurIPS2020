function [f,g,Hv,Gv] = gmmFunReg(w,X,C1,C2,phi,regFun)
if nargin==0
    tic;
    runTest
    toc;
    return;
end

w = shiftdim(w);

f_reg = 0;
g_reg = 0;
Hv_reg = @(v) 0;
if exist('regFun','var') && ~isempty(regFun)
    assert(isa(regFun,'function_handle'));
    if nargout >= 3
        [f_reg,g_reg,Hv_reg] = regFun(w);
    end
    if nargout == 2
        [f_reg,g_reg] = regFun(w);
    end
    if nargout == 1
        f_reg = regFun(w);
    end
end

[alpha,d_alpha,d2_alpha] = phi(w(1));
w = w(2:end);
d = length(w);
n = size(X,1);
assert( (d/2 == ceil(d/2)) && (d/2 == floor(d/2)) );
w1 = w(1:(d/2)); w2 = w((d/2+1):end);
assert( (size(X,2) == length(w1)) && (size(X,2) == length(w2)) );
[d1,d2] = size(C1);
assert(d1 >= d2)
assert(length(w1) == d2);
[d1,d2] = size(C2);
assert(d1 >= d2)
assert(length(w2) == d2);

W1 = C1*(X'-w1); % dd X n
t1 = sum(W1.*W1,1);
W2 = C2*(X'-w2); % dd X n
t2 = sum(W2.*W2,1);
c1 = abs(det(C1)); 
c2 = abs(det(C2));
c = max(c1,c2);
if alpha == 1
    m = t1;
    f1 = exp(-0.5*(t1-m))/(c1/c);
    f2 = zeros(size(f1));
else
    if alpha == 0
        m = t2;
        f2 = exp(-0.5*(t2-m))/(c2/c);
        f1 = zeros(size(f2));
    else
        m = min([t1;t2],[],1);
        f1 = exp(-0.5*(t1-m))/(c1/c);
        f2 = exp(-0.5*(t2-m))/(c2/c);
    end
end
f_f = sum( 0.5*m + log(c) - log( alpha*f1 + (1-alpha)*f2 ) ) ;
if nargout >= 2
    g0 = -d_alpha*sum( ( f1 - f2 )./( alpha*f1 + (1-alpha)*f2 ) );
    g1 = -( C1'*sum( W1 .*( alpha*f1./( alpha*f1 + (1-alpha)*f2 ) ), 2) );
    g2 = -( C2'*sum( W2 .*( (1-alpha)*f2./( alpha*f1 + (1-alpha)*f2 ) ), 2) );
    g_f = [g0;
           g1;
           g2];
end
if nargout == 3
    Hv_f = @(v) HVP(v,X,W1,W2,C1,C2,f1,f2,alpha,d_alpha,d2_alpha);
end

if nargout == 4
    Gv_f =@(v)  GVP(v,W1,W2,C1,C2,f1,f2,alpha,d_alpha);
end

f = f_f/n + f_reg;

if nargout >= 2
    g = g_f/n + g_reg;
end

if nargout == 3
    Hv = @(v)  Hv_f(v)/n + Hv_reg(v);
end

if nargout == 4
    Hv = []; % Just so Matlab does not complain!
    Gv =@(v)  Gv_f(v)/n + Hv_reg(v);
end

end

function Hv = HVP(v,X,W1,W2,C1,C2,f1,f2,alpha,d_alpha,d2_alpha)
% g0 = -d_alpha*sum((f1 - f2)./(alpha*f1 - f2*(alpha - 1)));
% g1 = -( C1'*sum( W1 .*( alpha*f1./( alpha*f1 + (1-alpha)*f2 ) ), 2) );
% g2 = -( C2'*sum( W2 .*( (1-alpha)*f2./( alpha*f1 + (1-alpha)*f2 ) ), 2) );
v0 = v(1);
v = v(2:end);
d = length(v);
f1 = shiftdim(f1);
f2 = shiftdim(f2);
assert(length(f1) == length(f2) && length(f1) == size(X,1));
assert( (d/2 == ceil(d/2)) && (d/2 == floor(d/2)) );
dd = size(X,2);
assert(dd == d/2);
v1 = v( 1:dd );
v2 = v( (dd+1):end );

vv1 = @(vv) W1'*( C1*vv ); 
vv1T = @(vv) C1'*( W1*vv ); 

vv2 = @(vv) W2'*( C2*vv ); 
vv2T = @(vv) C2'*( W2*vv );

F1 = f1./( alpha*f1 + (1-alpha)*f2 );
F2 = f2./( alpha*f1 + (1-alpha)*f2 );

H00_v0 = v0*sum( d_alpha*d_alpha*( ( F1 - F2 ).^2 ) - d2_alpha*( F1 - F2 ) );

H10_v0 = v0*d_alpha*[-( C1'*sum( W1 .*( F1.*F2 )', 2) );
                     -( C2'*sum( W2 .*( -F1.*F2 )', 2) )];

H01_v = dot(d_alpha*[-( C1'*sum( W1 .*( F1.*F2 )', 2) );
                     -( C2'*sum( W2 .*( - F1.*F2 )', 2) )], v);

H11_v1 =  ( ( C1'*sum( (C1*v1) .* ( alpha*F1 )' , 2 ) )  + ...
         ( -vv1T( vv1( v1 ) .* ( alpha*F1 ) ) ) + ...
          ( vv1T( vv1( v1 ) .* ( alpha*alpha*( F1.*F1 ) ) ) ) );

H22_v2 =  ( ( C2'*sum( (C2*v2) .* ( (1-alpha)*F2 )' , 2 ) )  + ...
         ( -vv2T( vv2( v2 ) .* ( (1-alpha)*F2 ) ) ) + ...
          ( vv2T( vv2( v2 ) .* ( (1-alpha)*(1-alpha)*( F2.*F2 ) ) ) ) );


H12_v2 =  ( vv1T( vv2( v2 ) .* ( alpha*(1-alpha)*( F1.*F2 ) ) ) );

H21_v1 =  ( vv2T( vv1( v1 ) .* ( alpha*(1-alpha)*( F1.*F2 ) ) ) );

Hv = [H00_v0 + H01_v;
      H10_v0 + [H11_v1 + H12_v2;
      H21_v1 + H22_v2]];
end

function Gv = GVP(v,W1,W2,C1,C2,f1,f2,alpha,d_alpha)
%g0 = -d_alpha*sum((f1 - f2)./(alpha*f1 - f2*(alpha - 1)));
%g1 = -( C1'*sum( W1 .*( alpha*f1./( alpha*f1 + (1-alpha)*f2 ) ), 2) );
%g2 = -( C2'*sum( W2 .*( (1-alpha)*f2./( alpha*f1 + (1-alpha)*f2 ) ), 2) );
v0 = v(1);
v = v(2:end);
d = length(v);

% d2f_alpha = shiftdim( ( (f1 - f2).^2 )./( ( alpha*f1 + (1-alpha)*f2 ).^2) );
% d2f_f1 = shiftdim( ( ( alpha*alpha*f1 .*f1 )./( ( alpha*f1 + (1-alpha)*f2 ).^2 ) ) );
% d2f_f2 = shiftdim( ( ( (1-alpha)*(1-alpha)*f2 .* f2 )./( alpha*f1 + (1-alpha)*f2 ).^2 ) );

sqrt_d2f = 1 ./ ( ( alpha*f1 + (1-alpha)*f2 ) );

assert( (d/2 == ceil(d/2)) && (d/2 == floor(d/2)) );
G0 = d_alpha*( ( f1 - f2 ) .* sqrt_d2f );
G1 = W1 .* ( ( alpha*f1 ) .* sqrt_d2f );
G2 = W2 .* ( ( (1-alpha)*f2 ) .* sqrt_d2f );

assert(length(f1) == length(f2) && length(f1) == size(G1,2) && length(f1) == size(G2,2));
dd = size(G1,1);
assert(dd == d/2);
assert(dd == size(G1,1));
v1 = v( 1:dd );
v2 = v( (dd+1):end );

G0v0 = sum( G0.*( G0*v0 ) );
G1v1 = C1'*( G1*(  G1' * ( C1 * v1 ) ) );
G2v2 = C2'*( G2*(  G2' * ( C2 * v2 ) ) );
Gv = [G0v0;
      G1v1; 
      G2v2];

end

function runTest
clc; clear all; close all; rehash;
n=100; dd = 10; 
X = randn(n,dd);
lambda = 0;
regularizer = @(w) regularizerL2(w, lambda);
C1 = randn(dd,dd);
C2 = rand(dd,dd);
derivativeTest(@(w) gmmFunReg(w,X,C1,C2,@(x) tanhFun(x,1),regularizer),[-1; randn(dd*2,1)])
end



% H00_v0 = v0*sum( d_alpha*d_alpha*( ( ( f1 - f2 )./( alpha*f1 + (1-alpha)*f2 ) ).^2 ) - d2_alpha*(f1 - f2)./( alpha*f1 + (1-alpha)*f2 ) );
% 
% H10_v0 = v0*d_alpha*[-( C1'*sum( W1 .*( ( f1.*f2 )./( alpha*f1 + (1-alpha)*f2 ).^2 )', 2) );
%                      -( C2'*sum( W2 .*( -(f1.*f2)./( alpha*f1 + (1-alpha)*f2 ).^2 )', 2) )];
% 
% H01_v = dot(d_alpha*[-( C1'*sum( W1 .*( (f1.*f2)./( alpha*f1 + (1-alpha)*f2 ).^2 )', 2) );
%                      -( C2'*sum( W2 .*( -(f1.*f2)./( alpha*f1 + (1-alpha)*f2 ).^2 )', 2) )], v);
% 
% H11_v1 =  ( ( C1'*sum( (C1*v1) .* ( alpha*f1./( alpha*f1 + (1-alpha)*f2 ) )' , 2 ) )  + ...
%          ( -vv1T( vv1( v1 ) .* ( alpha*f1 ./( alpha*f1 + (1-alpha)*f2 ) ) ) ) + ...
%           ( vv1T( vv1( v1 ) .* ( alpha*alpha*f1 .*f1 ./( ( alpha*f1 + (1-alpha)*f2 ).^2  ) ) ) ) );
% 
% H22_v2 =  ( ( C2'*sum( (C2*v2) .* ( (1-alpha)*f2./( alpha*f1 + (1-alpha)*f2 ) )' , 2 ) )  + ...
%          ( -vv2T( vv2( v2 ) .* ( (1-alpha)*f2 ./( alpha*f1 + (1-alpha)*f2 ) ) ) ) + ...
%           ( vv2T( vv2( v2 ) .* ( (1-alpha)*(1-alpha)*f2 .* f2 ./( ( alpha*f1 + (1-alpha)*f2 ).^2 ) ) ) ) );
% 
% 
% H12_v2 =  ( vv1T( vv2( v2 ) .* ( alpha*(1-alpha)*f2 .* f1 ./( ( alpha*f1 + (1-alpha)*f2 ).^2 ) ) ) );
% 
% H21_v1 =  ( vv2T( vv1( v1 ) .* ( alpha*(1-alpha)*f2 .* f1 ./( ( alpha*f1 + (1-alpha)*f2 ).^2 ) ) ) );