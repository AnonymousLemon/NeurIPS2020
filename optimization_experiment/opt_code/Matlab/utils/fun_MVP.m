function [ y ] =  fun_MVP( A, A_t, x,doTranspse )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if strcmp(doTranspse, 'notransp')
    y = A(x);
else
    y = A_t(x);
end
end

