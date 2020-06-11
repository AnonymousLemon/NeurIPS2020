function [f,g,Hv,Gv] = gmmFunReg(w,X,C1,C2,alpha,regularizer)
if nargin==0
    tic;
    runTest
    toc;
    return
end
assert(alpha >= 0 && alpha <=1);
w = shiftdim(w);
d = length(w);
assert( (d/2 == ceil(d/2)) && (d/2 == floor(d/2)) );
w1 = w(1:(d/2)); w2 = w((d/2+1):end);
assert( (size(X,2) == length(w1)) && (size(X,2) == length(w2)) );
[d1,d2] = size(C1);
assert(d1 >= d2)
assert(length(w1) == d2);
[d1,d2] = size(C2);
assert(d1 >= d2)
assert(length(w2) == d2);


W1 = C1*(X'-w1);
t1 = sum(W1.*W1,1);
W2 = C2*(X'-w2);
t2 = sum(W2.*W2,1);
m = min([t1;t2],[],1);
f1 = alpha*exp(-0.5*(t1-m));
f2 = (1-alpha)*exp(-0.5*(t2-m));
f_f = sum( 0.5*m - log( f1 + f2 ) ) ;
if nargout >= 2
    G1 = -( C1'*( C1*(X'-w1) ) ).*( f1./( f1 + f2 ) );
    G2 = -( C2'*( C2*(X'-w2) ) ).*( f2./( f1 + f2 ) ) ;
    g_f = [sum( G1 , 2 );
           sum( G2 , 2 )];
end
if nargout == 3
    Hv_f = @(v) HVP(v,X,w1,w2,C1,C2,f1,f2);
end

if nargout == 4
    Gv_f =@(v)  GVP(v,G1,G2,f1,f2);
end


if exist('regularizer','var') && ~isempty(regularizer)
    assert(isa(regularizer,'function_handle'));
    if nargout >= 3
        [f_reg,g_reg,Hv_reg] = regularizer(w);
    end
    if nargout == 2
        [f_reg,g_reg] = regularizer(w);
    end
    if nargout == 1
        f_reg = regularizer(w);
    end
end

f = f_f + f_reg;

if nargout >= 2
    g = g_f + g_reg;
end

if nargout == 3
    Hv = @(v)  Hv_f(v) + Hv_reg(v);
end

if nargout == 4
    Hv = []; % Just so Matlab does not complain!
    Gv =@(v)  Gv_f(v) + Hv_reg(v);
end


end

function Hv = HVP(v,X,w1,w2,C1,C2,f1,f2)
%G1 = - C1'*( C1*(X'-w1) ) ).*( f1./( f1 + f2 );
%G1 = - C2'*( C2*(X'-w2) ) ).*( f2./( f1 + f2 );
d = length(v);
f1 = shiftdim(f1);
f2 = shiftdim(f2);
assert(length(f1) == length(f2) && length(f1) == size(X,1));
assert( (d/2 == ceil(d/2)) && (d/2 == floor(d/2)) );
dd = size(X,2);
assert(dd == d/2);
v1 = v( 1:dd );
v2 = v( (dd+1):end );
Xmw1 = X'-w1; %dd X n
Xmw2 = X'-w2; % dd X n

vv1 = @(vv) Xmw1'*( C1'*( C1*vv ) ); 
vv1T = @(vv) C1'*( C1*( Xmw1*vv ) ); 

vv2 = @(vv) Xmw2'*( C2'*( C2*vv ) ); 
vv2T = @(vv) C2'*( C2*( Xmw2*vv ) );

H11_v1 =  sum( ( C1'*(C1*v1) ) .* ( f1./( f1 + f2 ) )' , 2 )  + ...
         -sum( vv1T( vv1( v1 ) .* ( f1 ./( f1 + f2 ) ) ) , 2 ) + ...
          sum( vv1T( vv1( v1 ) .* ( f1 .* f1 ./( ( f1 + f2 ).^2  ) ) ) , 2 );

H22_v2 =  sum( ( C2'*(C2*v2) ) .* ( f2./( f1 + f2 ) )' , 2 )  + ...
         -sum( vv2T( vv2( v2 ) .* ( f2 ./( f1 + f2 ) ) ) , 2 ) + ...
          sum( vv2T( vv2( v2 ) .* ( f2 .* f2 ./( ( f1 + f2 ).^2 ) ) ) , 2 );


H12_v2 =  sum( vv1T( vv2( v2 ) .* ( f2 .* f1 ./( ( f1 + f2 ).^2 ) ) ), 2 );

H21_v1 =  sum( vv2T( vv1( v1 ) .* ( f2 .* f1 ./( ( f1 + f2 ).^2 ) ) ), 2 );

Hv = [H11_v1 + H12_v2;
      H21_v1 + H22_v2];
end

function Gv = GVP(v,G1,G2,f1,f2)
%G1 = - C1'*( C1*(X'-w1) ) ).*( f1./( f1 + f2 );
%G1 = - C2'*( C2*(X'-w2) ) ).*( f2./( f1 + f2 );
d = length(v);
f1 = shiftdim(f1);
f2 = shiftdim(f2);
assert(length(f1) == length(f2) && length(f1) == size(G1,2) && length(f1) == size(G2,2));
assert( (d/2 == ceil(d/2)) && (d/2 == floor(d/2)) );
dd = size(G1,1);
assert(dd == d/2);
assert(dd == size(G1,1));
v1 = v( 1:dd );
v2 = v( (dd+1):end );
Gv1 = G1*( ( G1'* v1 ) );
Gv2 = G2*( ( G2'* v2 ) );
Gv = [Gv1; 
      Gv2];
end

function runTest
clc; clear all; close all; rehash;
n=100; dd = 10000; 
X = randn(n,dd);
lambda = 0;
regularizer = @(w) regularizerL2(w, lambda);
C1 = randn(dd,dd);
C2 = rand(dd,dd);
alpha = 0.2;
derivativeTest(@(w) gmmFunReg(w,X,C1,C2,alpha,regularizer),10*randn(dd*2,1))
end


% function Hv = HVP(v,X,w1,w2,C1,C2,f1,f2)
% %G1 = - C1'*( C1*(X'-w1) ) ).*( f1./( f1 + f2 );
% %G1 = - C2'*( C2*(X'-w2) ) ).*( f2./( f1 + f2 );
% d = length(v);
% assert( (d/2 == ceil(d/2)) && (d/2 == floor(d/2)) );
% [n,dd] = size(X);
% assert(dd == d/2);
% v1 = v( 1:dd );
% v2 = v( (dd+1):end );
% Xmw1 = X'-w1;
% Xmw2 = X'-w2;
% H11_v1 =  sum( ( repmat(C1'*(C1*v1),1,n) .* (f1./( f1 + f2 )) ) , 2 ) + ...
%               -( C1'*( C1* ( ( Xmw1 .* f1 ./( f1 + f2 ) ) * ( ( Xmw1' )*( C1'*( C1*v1 ) ) ) ) ) ) + ...
%                ( C1'*( C1* ( ( Xmw1 .* f1 .* f1 ./( ( f1 + f2 ).^2 ) ) * ( ( Xmw1' )*( C1'*( C1*v1 ) ) ) ) ) );
% 
% H22_v2 =  sum( ( repmat(C2'*(C2*v2),1,n) .* (f2./( f1 + f2 )) ) , 2 ) + ...
%               -( C2'*( C2*Xmw2 ) .* f2 ./( f1 + f2 ) ) * ( ( ( C2'*( C2*Xmw2 ) )'*v2 ) ) + ...
%                ( C2'*( C2*Xmw2 ) .* ( f2 .* f2 ./( ( f1 + f2 ).^2 ) ) ) * ( ( ( C2'*( C2*Xmw2 ) )'*v2 ) );
% 
% H12_v2 =  ( C1'*( C1*Xmw1 ) .* ( f2 .* f1 ./( ( f1 + f2 ).^2 ) ) ) *  ( ( C2'*( C2*Xmw2 ) )' * v2 );
% 
% H21_v1 =  ( C2'*( C2*Xmw2 ) .* ( f1 .* f2 ./( ( f1 + f2 ).^2 ) ) ) * ( ( C1'*( C1*Xmw1 ) )' * v1 );
% 
% Hv = [H11_v1 + H12_v2;
%       H21_v1 + H22_v2];
% end