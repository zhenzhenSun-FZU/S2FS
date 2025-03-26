function [U, U_i, S_b, S_w] = LDA_Regular(Xtrain, ytrain)
%UNTITLED8 此处显示有关此函数的摘要
%   此处显示详细说明
[d,m]=size(Xtrain);
S_b=zeros(d,d);
S_w=zeros(d,d);
sorted_target=sort(ytrain,1);
label=zeros(1,1);
N_i=zeros(1,1);
U = zeros(d,1);
label(1,1)=sorted_target(1,1);
j=1;
for i=2:m%%%%计算有几个类
   if sorted_target(i,1)~=label(j,1)
       j=j+1;
       label(j,1)=sorted_target(i,1);
   end
end
U_i = zeros(d,j);
number_class=j;
if number_class==2
   for i=1:m
       if(ytrain(i)==-1||ytrain(i)==0)
           ytrain(i)=2;
       end
   end
end
for i=1:number_class
      x=[];
      indx=find(ytrain==i);
      x=Xtrain(:,indx);
      [~,m1]=size(x);
      N_i(i,1)=m1;
      U_i(:,i) = mean(x,2);
      U(:,1) = U(:,1) + U_i(:,i)*m1;
end
U(:,1) = U(:,1)/m;

for i=1:number_class
    S_b(:,:)= S_b(:,:) +  N_i(i,1)*((U_i(:,i)-U(:,1))*(U_i(:,i)-U(:,1))');
    x=[];
    indx=find(ytrain==i);
    x=Xtrain(:,indx);
    [~,m1]=size(x);
    for j=1:m1
        S_w(:,:)= S_w(:,:) + (U_i(:,i)-x(:,j))*(U_i(:,i)-x(:,j))';
    end
end

