formula_lm<-function(y,covariates=c('GA.at.birth','PMA.at.birth','Gender'),prs,with_interaction=FALSE){
  if (length(covariates)==0){
    as.formula(paste(y,prs,sep = '~'))
  }
  else{
    if (with_interaction){
      as.formula(paste(y,paste(c(covariates,prs,paste(prs,covariates,sep='*')),collapse='+'),sep='~')) 
    }
    else{
      as.formula(paste(y,paste(c(covariates,prs),collapse='+'),sep='~')) 
    }
  }
}

PRS_scores=c('X1e.08','X1e.07','X1e.06','X1e.05','X0.0001','X0.001','X0.01','X0.05','X0.1','X0.5','X1')

calculate_p_val<-function(df,filename,perm_run=0,PRS_scores=c('X1e.08','X1e.07','X1e.06','X1e.05','X0.0001','X0.001','X0.01','X0.05','X0.1','X0.5','X1'),with_interaction=FALSE){
  #perform series of multiple linear regression with the response variable as diffusion value and predictors as prs score, pma, ga and gender
  for (prs in PRS_scores){
    for (i in 4:92){
      connection=colnames(df)[i]
      simple.regression<-lm(formula_lm(connection,prs=prs,with_interaction = FALSE),data=df)
      simple.regression.summary<-summary(simple.regression)
      temp_row<-data.frame(Connection=connection,
                           prs_pval=simple.regression.summary$coefficients[prs,4],
                           GA_pval=simple.regression.summary$coefficients['GA.at.birth',4],
                           PMA_pval=simple.regression.summary$coefficients['PMA.at.birth',4],
                           Gender_pval=simple.regression.summary$coefficients['Gender',4],
                           Adj_R_square=simple.regression.summary$adj.r.squared,
                           PRS_threshold=prs,
                           perm_run=perm_run
      )
      write.table(temp_row,file=filename,sep=',',col.names = !file.exists(filename),row.names=FALSE,append=T)
    }
  }
}

original_subset=read.csv('./original_subset_for_data_exploration_R_pca.csv')

permutation_n=500# set the permutation run
pb <- txtProgressBar(min = 0, max = permutation_n, style = 3)

for (i in 1:permutation_n){
  p=sample(1:nrow(original_subset))
  perm_subset=cbind(original_subset[,1:92],original_subset[p,93:103])
  # this line first applies R function sample to each of the 11 PRS columns of the original dataframe.
  # Once these columns are resampled, they are re-appended (cbind) to the rest of the orignal dataframe which contains diffusion values and PMA, GA and gender
  filename=paste('./log_pca/perm_table_pca_result_run_','.csv',sep=as.character(i))
  calculate_p_val(perm_subset,filename = filename,perm_run=i)
  setTxtProgressBar(pb, i)
}
close(pb)