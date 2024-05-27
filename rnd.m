clear all;
clc;

n = 1000;
A = randn(n,n) + 1i*randn(n,n);
%A = gallery('randhess',n);
[Q,~] = qr(A);


tic
H = (Q+Q') ; S = (Q-Q');
A_mu = (randn*H+randn*1i*S);
[U,D] = eig((A_mu+ A_mu') / 2);
toc


tic
H = (Q+Q') ; S = (Q-Q');
A_mu = (randn*H+randn*1i*S);
[~,D] = eig((A_mu+ A_mu') / 2);
toc


tic
[T,U] = schur(Q);
toc

tic
[P,H] = hess(Q);
toc

H = (Q+Q') /2 ;
tic
D = hess(Q);
toc

tic
[U,D] = hess(Q);
toc
