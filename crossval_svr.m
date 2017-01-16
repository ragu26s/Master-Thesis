function[soln_w,soln_reg,soln_tuberadius,soln_biassvr,soln_alphasvr,soln_alphastarsvr,soln_mse,fold_error] = crossval_svr(kfold,A_fold,b_fold,reg,tuberadius)

% %colors for lines pertaining to different alphas
% cl=hsv(length(tuberadius));

%to get the indices for split up
rng(1234);
indices=crossvalind('Kfold',size(A_fold,1),kfold);

%to hold the training set
A_train=[];
b_train=[];

 %to hold the validation set
 err_tr=zeros(size(A_fold,1)*(kfold-1)/kfold,length(reg)*length(tuberadius));
 err_val=zeros(size(A_fold,1)/kfold,length(reg)*length(tuberadius));
fold_error=zeros(kfold,length(reg)*length(tuberadius));
total_error=zeros(1,length(reg)*length(tuberadius));

%To hold MSE for training set
err_trf=zeros(size(A_fold,1),length(reg)*length(tuberadius));

%To hold MSE for validation set
err_valf=zeros(size(A_fold,1),length(reg)*length(tuberadius));

%Combinatorion of training sets to be combined all but one folds
seq=combnk(1:kfold,kfold-1);

    for i=1:size(seq,1)
        A_train=[];
        b_train=[];
        for j=1:size(seq,2)
           
            %Training sets to be combined
            A_temp=A_fold(indices==seq(i,j),:);
            b_temp=b_fold(indices==seq(i,j),:);
            
            %Append
            A_train=[A_train;A_temp];
            b_train=[b_train;b_temp];
        end
       
%         %Taking the validation set
         diff_index=setdiff(1:kfold,seq(i,:));
        A_val=A_fold(indices==diff_index,:);
        b_val=b_fold(indices==diff_index,:);
%     
%         %Operation
         [wsvrcv,alphasvrcv,alphastarsvrcv,esvrcv,estarsvrcv,biassvrcv]=svr(A_train,b_train,reg,tuberadius);
         %[err_trf]=svr_error(A_train,b_train,wsvrcv,biassvrcv,tuberadius,reg);
         [err_valf]=svr_error(A_val,b_val,wsvrcv,biassvrcv,tuberadius,reg);
%         
%         %Average out the appended MSE
         %err_tr=err_tr+(err_trf)/kfold; 
         err_val=err_val+(err_valf);
         
         %calculate MSE
         for m=1:length(reg)
             for k=1:length(tuberadius)
                 mse_val((k-1)*length(reg)+m)=norm(err_valf(:,(k-1)*length(reg)+m),2);
                 %disp(mse_val((k-1)*length(reg)+m));
             end
         end
         fold_error(i,:)=mse_val;
         
     
%         [min_mse,min_ind]=min(mse_val);
%         quo=fix(min_ind/length(reg));
%         rem=mod(min_ind,length(reg));
%         if rem==0  
%             rem=length(reg);
%             quo=quo-1;
%         end
%         soln_reg(i)=reg(rem);
%         soln_tuberadius(i)=tuberadius(quo+1);
%         soln_w(:,i)=wsvrcv(:,min_ind);
%         soln_alphasvr(:,i)=alphasvrcv(:,min_ind);
%         soln_alphastarsvr(:,i)=alphastarsvrcv(:,min_ind);
%         soln_mse(:,i)=min_mse;
%         soln_biassvr(:,i)=biassvrcv(min_ind);
         
    end
    for i=1:kfold
        total_error=total_error+fold_error(i,:);
    end
    total_error=total_error/kfold;
    [min_mse,min_ind]=min(total_error);
        quo=fix(min_ind/length(reg));
        rem=mod(min_ind,length(reg));
        if rem==0  
            rem=length(reg);
            quo=quo-1;
        end
    soln_reg=reg(rem);
    soln_tuberadius=tuberadius(quo+1);
    soln_w=wsvrcv(:,min_ind);
    soln_alphasvr=alphasvrcv(:,min_ind);
    soln_alphastarsvr=alphastarsvrcv(:,min_ind);
    soln_mse=min_mse;
    soln_biassvr=biassvrcv(min_ind);
%     %PLOT
%     figure;
%     for j=1:length(tuberadius)
%         for i=1:length(reg)    
%             a(i)=norm(err_tr(:,(j-1)*length(C)+i),2);
%         end
%         semilogx(reg,a,'Color',cl(j,:));
%         hold on;
%         legendInfo{j} = ['Tube Radius= ' num2str(tuberadius(j))];
%     end 
%     hold off;
%     xlabel('Regularization Parameter')
%     ylabel('l2-norm of error')
%     title('Error for 5-fold Cross Validation - Training SVR');
%     grid on;
%     legend(legendInfo);
%     hold off;
%  
%  %PLOT
%  figure;
%     for j=1:length(tuberadius)
%         for i=1:length(reg)    
%             a(i)=norm(err_val(:,(j-1)*length(C)+i),2);
%         end
%         semilogx(reg,a,'Color',cl(j,:));
%         hold on;
%         legendInfo{j} = ['Tube Radius= ' num2str(tuberadius(j))];
%     end 
%     hold off;
%     xlabel('Regularization Parameter')
%     ylabel('l2-norm of error')
%     title('Error for 5-fold Cross Validation - Validation SVR');
%     grid on;
%     legend(legendInfo);
%     hold off;
end