function[soln_fucmse,soln_fcmse,total_ucmse,total_cmse,soln_reg,soln_tunein,l_uctime,l_ctime] = crossval_lasso_hub(kfold,A_fold,b_fold,reg,tunein)
%5-fold CV implemented for LASSO
% soln_xc=zeros(size(A_fold,2),kfold);
% soln_xuc=zeros(size(A_fold,2),kfold);
% soln_reg=zeros(kfold,1);
% soln_tunein=zeros(kfold,1);
% soln_cmse=zeros(kfold,1);
% soln_ucmse=zeros(kfold,1);
soln_fcmse=zeros(length(reg),kfold);
soln_fucmse=zeros(length(reg),kfold);
total_ucmse=zeros(length(reg),1);
total_cmse=zeros(length(reg),1);
%to get the indices for split up
rng(1234);
indices=crossvalind('Kfold',size(A_fold,1),kfold);
l_uctime=0;
l_ctime=0;
%to hold the training set
A_train=[];
b_train=[];

%to hold the validation set
A_val=[];
b_val=[];
%
% %To hold MSE for training set
% mse_ctrain=zeros(length(reg),1);
% mse_uctrain=zeros(length(reg),1);

%To hold MSE for validation set
mse_cval=zeros(length(reg),1);
mse_ucval=zeros(length(reg),1);

%Combination of training sets to be combined all but one folds
seq=combnk(1:kfold,kfold-1);

for i=1:size(seq,1)
    %to hold the training set
    A_train=[];
    b_train=[];
    
    %to hold the validation set
    A_val=[];
    b_val=[];
    for j=1:size(seq,2)
        %Training sets to be combined
        A_temp=A_fold(indices==seq(i,j),:);
        b_temp=b_fold(indices==seq(i,j),:);
        
        %Append
        A_train=[A_train;A_temp];
        b_train=[b_train;b_temp];
    end
    
    %Taking the validation set
    diff_index=setdiff(1:kfold,seq(i,:));
    A_val=A_fold(indices==diff_index,:);
    b_val=b_fold(indices==diff_index,:);
    n_val=size(A_val,1);
    %Operation
    [xc_l,xuc_l,~,~,~,uctime,ctime]=lasso_op_hub(A_train,b_train,reg,tunein);%OPS
    l_uctime=l_uctime+uctime;
    l_ctime=l_ctime+ctime;
    %Record details for min MSE pertaining to Unonstrained LASSO
    [mse_luc]=msecomp([ones(n_val,1) A_val],b_val,xuc_l,reg);%OPS
    soln_fucmse(:,i)=mse_luc;
    total_ucmse=total_ucmse+mse_luc;
    %         [min_mse,min_ind]=min(mse_luc);
    %         soln_xuc(:,i)=xuc_l(:,min_ind);
    %         soln_ucmse(i)=min_mse;
    %         soln_reg(i)=reg(min_ind);
    
    %Record details for min MSE pertaining to Constrained LASSO
    [mse_lc]=msecomp([ones(n_val,1) A_val],b_val,xc_l,tunein);
    soln_fcmse(:,i)=mse_lc;
    total_cmse=total_cmse+mse_lc;
    %         [min_mse,min_ind]=min(mse_lc);
    %         soln_xc(:,i)=xc_l(:,min_ind);
    %         soln_cmse(i)=min_mse;
    %         soln_tunein(i)=tunein_lasso(min_ind);
    
    
    %Average out the appended MSE
end
total_ucmse=total_ucmse/kfold;
total_cmse=total_cmse/kfold;
[~,min_ind]=min(total_ucmse);
soln_reg=reg(min_ind);
[~,min_ind]=min(total_cmse);
soln_tunein=tunein(min_ind);


%     %PLOT
%     figure;
%     semilogx(reg,mse_uctrain,'ro');
%     hold on;
%     semilogx(reg,mse_ucval,'b.');
%     xlabel('Regularization Parameter');
%     ylabel('MSE');
%     title('MSE for 5-fold Cross Validation - Unconstrained LASSO');
%     grid on;
%     legend('Average of Training Set','Average of Validation set');
%
%     %PLOT
%     figure;
%     plot(tunein_lasso,mse_ctrain,'ro');
%     hold on;
%     plot(tunein_lasso,mse_cval,'b.');
%     xlabel('Tune-in Parameter');
%     ylabel('MSE');
%     title('MSE for 5-fold Cross Validation - Constrained LASSO');
%     grid on;
%     legend('Average of Training Set','Average of Validation set');
end