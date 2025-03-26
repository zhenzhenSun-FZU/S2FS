clc; clear; 
addpath(genpath('.\'))
DataName={'breast3'};
for d=1:length(DataName)
    dataset=DataName{d};
    datapath=strcat('./datasets/',dataset);
    load(datapath);
     
     [num_data,num_fea] = size(X);
     randorder = randperm(num_data);
     cv_num = 5;
     %parameters setting
     alphaCandi=[10^-6,10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2];
     lambdaCandi=[10^-6,10^-5,10^-4,10^-3,10^-2];
     mu = 0.1;
     rho = 1.05;
     
     for cv=1:cv_num
         result_path=strcat(dataset,'/','cv',num2str(cv));
         mkdir(result_path);
         [cv_train_data,cv_test_data,cv_train_target,cv_test_target] = selectsamples(X',Y);
         [U, U_i, S_b, S_w] = LDA_Regular(cv_train_data, cv_train_target);
          for  a=1:9
             for l=1:5
                 [W,obj,time] = S2FS_v2(cv_train_data, cv_train_target, S_b, S_w,alphaCandi(a),lambdaCandi(l),mu,rho);
                 dumb= sum(W.*W,2);
                 [~,ranked]=sort(dumb,'descend');
                 selectedIndices=ranked(1:20);
                  DataTrain=cv_train_data(selectedIndices,:);
                  DataTest=cv_test_data(selectedIndices,:);
                  acc_soft_20(a,l) = sofmax_classfier( DataTrain, cv_train_target, DataTest, cv_test_target);
                  acc_KNN_20(a,l) =  KNN(DataTrain,cv_train_target, DataTest, cv_test_target,5);
              
                  selectedIndices=ranked(1:40);
                  DataTrain=cv_train_data(selectedIndices,:);
                  DataTest=cv_test_data(selectedIndices,:);
                  acc_soft_40(a,l) = sofmax_classfier( DataTrain, cv_train_target, DataTest, cv_test_target);
                  acc_KNN_40(a,l) =  KNN(DataTrain,cv_train_target, DataTest, cv_test_target,5);     
            end 
         end
         save_path=strcat(result_path,'/','result.mat');
         save(save_path,'acc_soft_20','acc_KNN_20','acc_soft_40','acc_KNN_40');
     end                        
end
