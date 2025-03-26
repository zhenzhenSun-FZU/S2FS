% This is an example file on how the S2FS [1] program could be used.

% [1] Sun Z, Chen Z, Liu J, et al.
% Multi-class feature selection via Sparse Softmax with a discriminative regularization[J]. 
%International Journal of Machine Learning and Cybernetics, 2025,16(1):159-172.

clc;clear;
addpath(genpath('.\'))
dataname = 'brain';
datapath=strcat('./datasets/',dataname,'.mat');
load(datapath);
%parameters setting
alphaCandi=[10^-6,10^-5,10^-4,10^-3,10^-2,10^-1,10^0,10^1,10^2];
lambdaCandi=[10^-6,10^-5,10^-4,10^-3,10^-2];
 
 cv_num = 5;
 for cv=1:5
    fprintf('Data processing, Cross validation: %d\n', cv);
     result_path=strcat(dataname,'/','cv',num2str(cv));
     mkdir(result_path);
    % Training set and test set for the i-th fold
    [cv_train_data,cv_test_data,cv_train_target,cv_test_target] = selectsamples(X',Y);
    [U, U_i, S_b, S_w] = LDA_Regular(cv_train_data, cv_train_target);
    % Running the S2FS procedure for feature selection
     for  a=1:9
          for l=1:5
              t0 = clock;
              [W,obj] = S2FS_v1(cv_train_data, cv_train_target, S_b, S_w,alphaCandi(a),lambdaCandi(l));
              time = etime(clock, t0);
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

 
 