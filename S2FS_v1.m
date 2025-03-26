function [W, obj] = S2FS_v1(X, labels, Scatter_b, Scatter_w, gamma,lambda_tgt)
% Proximal Gradient method to solve Softmax + gamma*(tr(W'SwW)+||W'SbW -I||_2) + lambda*||W||_2,0
%

if(nargin<6)
    lambda_tgt=1e-4;%lambda����Сֵ
end
lambda=1;
numCases=size(X,2);
Y = full(sparse(labels, 1:numCases, 1));
W=zeros(size(X,1),size(Y,1));
obj = [];
Innermaxit=20;
MaxIter=50;
X = X*diag(1./sqrt(sum(X.*X)));%��һ��
L=max(sum(X.*X,1));
eta=0.5;%lambda��ѧϰ�ʲ���
alpha=1.0/L;
beta=1e-1;%W�Ĳ���
tol=0.1;
N_stages = floor(log(lambda/lambda_tgt)/log(1.0/eta));%floor:����ȡ��

[fw,gw]=func(W,Y,X,Scatter_b,Scatter_w,gamma);%���fw�����ݶ�gw
obj1 = fw + lambda*L0_norm(W);
obj = [obj,obj1];

for i=1:N_stages-1
    lambda=lambda*eta;
    tol=max(tol/10, 1e-5);
        %������L��ȡֵ
    for t=1:Innermaxit
          W1=W-alpha*gw;
          W1=hard_mapping(W1,2*lambda*alpha);
          [fw1,gw1]=func(W1,Y,X,Scatter_b,Scatter_w,gamma);%���fw1�����ݶ�gw1
          dw=W1-W;
          %�ж�L�����������L�Ѿ�������������L����Ҫ�ٱ仯
          if fw+lambda*L0_norm(W)>=fw1+lambda*L0_norm(W1)+0.5*beta*norm(dw,'fro')^2
                break;
          end
          alpha=alpha*0.5;
     end
     %calcaulate loss  
     obj1 = fw1+ lambda*L0_norm(W1);
     obj = [obj,obj1];
     W=W1;
     fw=fw1;
     gw=gw1;
end
lambda=lambda_tgt;
for j=1:MaxIter
        %������L��ȡֵ
     for t=1:Innermaxit
          W1=W-alpha*gw;
          W1=hard_mapping(W1,2*lambda*alpha);
          [fw1,gw1]=func(W1,Y,X,Scatter_b,Scatter_w,gamma);%���fw1�����ݶ�gw1
          dw=W1-W;
          %�ж�L�����������L�Ѿ�������������L����Ҫ�ٱ仯
          if fw+lambda*L0_norm(W)>=fw1+lambda*L0_norm(W1)+0.5*beta*norm(dw,'fro')^2
               break;
          end
          alpha=alpha*0.5;
      end
      %calcaulate loss  
      obj2 = fw1+ lambda*L0_norm(W1);
      obj = [obj,obj2];
      norm_W=norm(W,'fro');
      if norm_W==0
          norm_W=eps;
      end
      if  (norm(obj2-obj1)/norm(obj1)<1e-6) 
           break;
      end
      W=W1;
      fw=fw1;
      gw=gw1;
      obj1=obj2;
end
end
%��fx�����ݶ�
function [fw,gw]=func(W,Y,X,Scatter_b, Scatter_w,gamma)
N = size(X,2);
K = size(Y,1);
Z = zeros(K,N);
for n = 1:N
    sum_exp = 0;
    for k = 1:K
        sum_exp = sum_exp + exp(W(:,k)'*X(:,n));
    end
    for k = 1:K
        Z(k,n) = exp(W(:,k)'*X(:,n))/sum_exp;
    end
end

J = 0;
for n = 1:N
    for k = 1:K
        J = J + Y(k,n)*log(Z(k,n));
    end
end

fw = -J+gamma*(trace(W'*Scatter_w*W)+trace((W'*Scatter_b*W-eye(K))'*(W'*Scatter_b*W-eye(K))));

gw = zeros(size(X,1),size(Y,1));
for k = 1:K
    for n = 1:N
        gw(:,k) = gw(:,k) + (Z(k,n)-Y(k,n))*X(:,n);
    end
end
gw(:,:) = gw(:,:) + gamma*((Scatter_w*W + Scatter_w'*W) + 2*(Scatter_b*W + Scatter_b'*W)*(W'*Scatter_b*W-eye(K)));

end

%��W���㷶��
function num=L0_norm(W)
num=0;
for i=1:size(W,1)
    if(nnz(W(i,:)~=0))%���D�ĵ�i���������ķ���Ԫ�ظ���������0����ȡ1
        num=num+1;
    end
end
end

function x=hard_mapping(z,lambda)
[row,col]=size(z);%rowΪz��������colΪz������
x=zeros(row,col);%���⿪һ�������
alpha=sqrt(lambda);%lambda������
for i=1:row %��Ŀ������е�ÿһ�У�����ģ��<=�ٽ�ֵ����ȡ��
    if norm(z(i,:))>alpha
        x(i,:)=z(i,:);
    else
        x(i,:)= 0;
    end
end

end


