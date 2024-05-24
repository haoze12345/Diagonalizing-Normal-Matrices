clear all;
clc;

n = 500;
A = randn(n,n) + 1i*randn(n,n);
%A = gallery('randhess',n);
[Q,~] = qr(A);


tic
H = (Q+Q') ; S = (Q-Q');
A_mu = (randn*H+randn*1i*S);
[U,D] = eig((A_mu+ A_mu') / 2);
U'*Q*U;
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