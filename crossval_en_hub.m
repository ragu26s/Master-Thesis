function[total_ucmse,soln_ucmse,soln_reg,soln_alphauc,total_cmse,soln_cmse,soln_tunein,soln_alphac,fold_ucmse,fold_cmse,en_uctime,en_ctime] = crossval_en_hub(kfold,A_fold,b_fold,reg,alpha,tunein)
%5-fold CV implemented for EN
% soln_xc=zeros(size(A_fold,2),kfold);
% soln_xuc=zeros(size(A_fold,2),kfold);
% soln_reg=zeros(kfold,1);
% soln_tunein=zeros(kfold,1);
% soln_cmse=zeros(kfold,1);
% soln_ucmse=zeros(kfold,1);
% soln_alphauc=zeros(kfold,1);
% soln_alphac=zeros(kfold,1);
fold_ucmse=zeros(length(alpha),length(reg),kfold);
fold_cmse=zeros(length(alpha),length(reg),kfold);
total_ucmse=zeros(length(alpha),length(reg));
total_cmse=zeros(length(alpha),length(reg));

%colors for lines pertaining to different alphas
cl=hsv(length(alpha));

%to get the indices for split up
rng(1234);
indices=crossvalind('Kfold',size(A_fold,1),kfold);

en_uctime=0;
en_ctime=0;
%to hold the training set
A_train=[];
b_train=[];

%to hold the validation set
A_val=[];
b_val=[];

%To hold MSE for training set
mse_ctrain=zeros(length(alpha),length(reg));
mse_uctrain=zeros(length(alpha),length(reg));

%To hold MSE for validation set
mse_cval=zeros(length(alpha),length(reg));
mse_ucval=zeros(length(alpha),length(reg));

%Combinatorion of training sets to be combined all but one folds
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
    n_val=size(b_val,1);
    
    %Operation
    [xc_en,xuc_en,~,~,uctime,ctime]=en_op_hub(A_train,b_train,reg,alpha,tunein);%OPS
    en_uctime=en_uctime+uctime;
    en_ctime=en_ctime+ctime;
    [mse_enuc]=msecomp_en([ones(n_val,1) A_val],b_val,xuc_en,reg,alpha);%OPS
    [mse_enc]=msecomp_en([ones(n_val,1) A_val],b_val,xc_en,reg,alpha);%OPS
    %         temp_mse=Inf(1);
    %         for k=1:length(alpha)
    %             [min_mse,min_ind]=min(mse_luc(k,:));
    %             if (min_mse<temp_mse)
    %                 soln_xuc(:,i)=xuc_en(:,min_ind,k);
    %                 soln_ucmse(i)=min_mse;
    %                 soln_reg(i)=reg(min_ind);
    %                 soln_alphauc(i)=alpha(k);
    %
    %             end
    %         end
    %         temp_mse=Inf(1);
    %         [mse_uc]=msecomp_en(A_val,b_val,xc_en,reg,alpha);
    %         for k=1:length(alpha)
    %             [min_mse,min_ind]=min(mse_uc(k,:));
    %             if (min_mse<temp_mse)
    %                 soln_xc(:,i)=xc_en(:,min_ind,k);
    %                 soln_cmse(i)=min_mse;
    %                 soln_tunein(i)=tunein_en(min_ind);
    %                 soln_alphac(i)=alpha(k);
    %             end
    %         end
    fold_ucmse(:,:,i)=mse_enuc;
    fold_cmse(:,:,i)=mse_enc;
    total_ucmse=total_ucmse+mse_enuc;
    total_cmse=total_cmse+mse_enc;
    
end
total_ucmse=total_ucmse/5;
total_cmse=total_cmse/5;
temp_mse=Inf(1);
for k=1:length(alpha)
    [min_mse,min_ind]=min(total_ucmse(k,:));
    if (min_mse<temp_mse)
        soln_ucmse=min_mse;
        soln_reg=reg(min_ind);
        soln_alphauc=alpha(k);
    end
end

temp_mse=Inf(1);
for k=1:length(alpha)
    [min_mse,min_ind]=min(total_cmse(k,:));
    if (min_mse<temp_mse)
        soln_cmse=min_mse;
        soln_tunein=tunein(min_ind);
        soln_alphac=alpha(k);
    end
end

%     %PLOT
%     figure;
%     for k=1:length(alpha)
%         semilogx(reg,mse_uctrain(k,:),'Color',cl(k,:));
%         hold on;
%         legendInfo{k} = ['Alpha= ' num2str(alpha(k))];
%     end
%     xlabel('Regularization Parameter')
%     ylabel('MSE')
%     title('MSE for 5-fold Cross Validation - Unconstrained Elastic Net Training');
%     grid on;
%     legend(legendInfo);
%     hold off;
%
%     figure;
%     for k=1:length(alpha)
%         hold on;
%         semilogx(reg,mse_ucval(k,:),'Color',cl(k,:));
%         hold on;
%         legendInfo{k} = ['Alpha= ' num2str(alpha(k))];
%     end
%     xlabel('Regularization Parameter')
%     ylabel('MSE')
%     title('MSE for 5-fold Cross Validation - Unconstrained Elastic Net Validation');
%     grid on;
%     legend(legendInfo);
%     hold off;

%
%     %PLOT
%     figure;
%     for k=1:length(alpha)
%         plot(tunein_en(k,:),mse_ctrain(k,:),'Color',cl(k,:));
%         hold on;
%         legendInfo{k} = ['Alpha= ' num2str(alpha(k))];
%     end
%     xlabel('Tune-in Parameter')
%     ylabel('MSE')
%      title('MSE for 5-fold Cross Validation - Constrained Elastic Net Training');
%     grid on;
%     legend(legendInfo);
%     hold off;
%
%         %PLOT
%     figure;
%     for k=1:length(alpha)
%         plot(tunein_en(k,:),mse_cval(k,:),'Color',cl(k,:));
%         hold on;
%         legendInfo{k} = ['Alpha= ' num2str(alpha(k))];
%     end
%     xlabel('Tune-in Parameter')
%     ylabel('MSE')
%      title('MSE for 5-fold Cross Validation - Constrained Elastic Net Validation');
%     grid on;
%      legend(legendInfo);
%     hold off;
end