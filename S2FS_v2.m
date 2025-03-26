function [W,obj,time] = S2FS_v2(X, labels , Sb, Sw, alpha, lambda, mu,rho)
%S2FS ADMM to solve min_{W'SbW = I}  Softmax + alpha*tr(W'SwW) + lambda*||W||_2,0
% X: d*n training data matrix, each column is a data point
% Y: c*n label matrix
% Sb: between-class scatter matrix
% Sw: within-class scatter matrix
% alpha, lambda: parameter
% mu, rho: parameters in the ADMM optimization method

[dim,numCases]=size(X);
Y = full(sparse(labels, 1:numCases, 1));
%% parameters setting
MaxIter = 30;
minimumLossMargin = 1e-4;
num_class = size(Y,1);
obj = [];

%% Initialize
Lambda1 = zeros(dim, num_class);
Lambda2 = zeros(dim, num_class);
W = randn(dim, num_class);
M = W;
O = W;

err=1; iter=1;
tic;
while (err > minimumLossMargin && iter<= MaxIter)
    inmu = 1/mu;
    %update W
    D_weight = diag( 0.5./sqrt(sum(W.*W,2)+eps));
    H_Sb = (Sb + D_weight)^.5;
    A = real(H_Sb \(eye(dim)+inmu*alpha*Sw)/H_Sb );
    B = real(H_Sb \(M + O + inmu*(Lambda1+Lambda2)));
    Q = gpi(A,B,1);
    W = H_Sb\Q;
    
    %update M
    V = W+inmu*Lambda1;
    M = hard_mapping(V,2*lambda*inmu);
    
    %update O
    L = W+inmu*Lambda2;
    options.maxIter = 100;
    O = softmax_least_square_Train(O, X, Y,L, mu, options);
    
    Lambda1 = Lambda1 + mu*(W-M);
    Lambda2 = Lambda2 + mu*(W-O);   
    mu = min(10^10,rho*mu);
    
    loss = softmax_cost(X, Y, W);
    obj(iter) = loss+alpha*trace(W'*Sw*W) + lambda*L0_norm(M);
    
    if iter>5
    err = (obj(iter-1)-obj(iter))/abs(obj(iter-1));
    end
    if err < 0
        break;
    end
    
   iter = iter+1;
end
time = toc;
end

%Title: A generalized power iteration method for solving quadratic problem on the Stiefel manifold

%% Authors: Feiping Nie, Rui Zhang, and Xuelong Li.

%Citation: SCIENCE CHINA Information Sciences 60, 112101 (2017); doi: 10.1007/s11432-016-9021-9

%View online: http://engine.scichina.com/doi/10.1007/s11432-016-9021-9

%View Table of Contents:http://engine.scichina.com/publisher/scp/journal/SCIS/60/11

%Published by the Science China Press

%% Generalized power iteration method (GPI) for solving min_{W??W=I}Tr(W??AW-2W^TB)

%Input: A as any symmetric matrix with dimension m*m; B as any skew matrix with dimension m*k,(m>=k);

%In particular, s can be chosen as 1 or 0, which stand for different ways of determining relaxation parameter alpha. 

%i.e. 1 for the power method and 0 for the eigs function.

%Output: solution W and convergent curve.

function W=gpi(A,B,s)

if nargin<3
    s=1;

end
[m,k]=size(B);

if m<k 
    disp('Warning: error input!!!');
    W=null(m,k);
    return;
end
A=max(A,A');
if s==0
    alpha=abs(eigs(A,1));
else if s==1
    ww=rand(m,1);
    for i=1:10
        m1=A*ww;
        q=m1./norm(m1,2);
        ww=q;
    end
    alpha=abs(ww'*A*ww);
    else disp('Warning: error input!!!');
         W=null(m,k);
         return;
    end
end

err1=1;t=1;
W=orth(rand(m,k));
A_til=alpha.*eye(m)-A;

while t<5
    M=2*A_til*W+B;
    [U,~,V]=svd(M,'econ');
    W=U*V';
    obj(t)=trace(W'*A*W-W'*B);
    if t>=2
        err1=abs(obj(t-1)-obj(t));
    end
        t=t+1;
end
end

function x=hard_mapping(z,lambda)
[row,col]=size(z);%row为z的行数，col为z的列数
x=zeros(row,col);%另外开一个矩阵存
alpha=sqrt(lambda);%lambda开更号
for i=1:row %对目标矩阵中的每一行，若是模长<=临界值，则取零
    if norm(z(i,:))>alpha
        x(i,:)=z(i,:);
    else
        x(i,:)= 0;
    end
end

end
%求W的零范数
function num=L0_norm(W)
num=0;
for i=1:size(W,1)
    if(nnz(W(i,:)~=0))%如果D的第i个列向量的非零元素个数不等于0，则取1
        num=num+1;
    end
end
end

function cost=softmax_cost(X, Y, theta)
numCases = size(X, 2);
M = theta'*X;
M = bsxfun(@minus,M,max(M, [], 1));   
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
cost = -1/numCases * Y(:)' * log(p(:));
end

