function [train_data,test_data ,trls, ttls] = selectsamples(data,labels)
train_data=[];
test_data=[];
train_index=[];
test_index=[];
trls=[];
ttls=[];
[~,m]=size(data);
sorted_target=sort(labels,1);
label=zeros(1,1);
label(1,1)=sorted_target(1,1);
j=1;
for i=2:m%%%%计算有几个类
   if sorted_target(i,1)~=label(j,1)
       j=j+1;
       label(j,1)=sorted_target(i,1);
   end
end
number_class=j;
if number_class==2
   for i=1:m
       if(labels(i)==-1||labels(i)==0)
           labels(i)=2;
       end
   end
end
for i=1:number_class
      x=[];
      indx=find(labels==i);
      x=data(:,indx);
      T=labels(indx);
      [~,m1]=size(x);
       if m1<3
          t=ceil(m1/2);
          else t=ceil(m1*4/5);
      end
        randnum=randperm(m1);
        ind=randnum(1:t);
        train_data=[train_data,x(:,ind)];
        trls=[trls;T(ind)];
        train_index=[train_index;indx(ind)];
        x(:,ind)=[];
        T(ind)=[];
        indx(ind)=[];
        test_data=[test_data,x];
        ttls=[ttls;T]; 
        test_index=[test_index;indx];
end

    



