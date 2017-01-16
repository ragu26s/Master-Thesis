function[soln_reg,soln_alphasvr,fold_error,soln_biassvr] = crossval_lssvrdual_lin(kfold,A_fold,b_fold,reg,tuberadius,ktype)

% %colors for lines pertaining to different alphas
% cl=hsv(length(tuberadius));

%to get the indices for split up
rng(1234);
indices=crossvalind('Kfold',size(A_fold,1),kfold);

%to hold the training set
A_train=[];
b_train=[];

fold_error=zeros(length(reg)*length(tuberadius),kfold);
total_error=zeros(length(reg)*length(tuberadius),1);

%  %to hold the validation set
%  err_tr=zeros(size(A_fold,1)*(kfold-1)/kfold,length(reg)*length(tuberadius));
%  err_val=zeros(size(A_fold,1)/kfold,length(reg)*length(tuberadius));

% %To hold MSE for training set
% err_trf=zeros(size(A_fold,1),length(reg)*length(tuberadius));
% 
% %To hold MSE for validation set
% err_valf=zeros(size(A_fold,1),length(reg)*length(tuberadius));

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
        %Parameter Setting
         if (strcmp(ktype,'gaussian'))
             gaussian_par=gaussian_hue(A_train);
         else
                gaussian_par=0;
         end  
        
%         %Operation
         [fdualalpha,fdualbiassvr]=lssvrdual(A_train,b_train,reg,ktype,gaussian_par);
         K=kernelmatrix_test(ktype,A_train,A_val,gaussian_par);
         err_trf=lssvrdual_error(K,b_val,fdualalpha,fdualbiassvr,reg);
        
%             
         %calculate MSE
         for m=1:length(reg)
             
            mse_val(m)=sqrt((sum(err_trf(:,m).^2))/size(err_trf(:,m),1));
         end

        fold_error(:,i)=mse_val;
            
    end
     for i_temp=1:kfold
        total_error=total_error+fold_error(:,i_temp);
    end
    total_error=total_error/kfold;
    
    
    
    [min_mse,min_ind]=min(total_error);
%         quo=fix(min_ind/length(reg));
%         rem=mod(min_ind,length(reg));
%         if rem==0  
%             rem=length(reg);
%             quo=quo-1;
%         end
        soln_reg=reg(min_ind);
        %soln_tuberadius=tuberadius(quo+1);
        %soln_w(:,i)=fsvrw_d(:,min_ind);
        soln_alphasvr=fdualalpha(:,min_ind);
        soln_biassvr=fdualbiassvr(min_ind);
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