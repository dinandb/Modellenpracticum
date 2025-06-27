
getMaxVals <- function(data, colname="z_wf") {
  # colname moet een string zijn
  
  maxvals <- c()
  
  new_data <- c()
  indices <- c()
  
  for (i in 3:(length(data[[colname]])-2)){
    if (((data[[colname]][i-2])>(data[[colname]][i-1]) & (data[[colname]][i-1])>(data[[colname]][i]) 
        & (data[[colname]][i])<(data[[colname]][i+1]) & (data[[colname]][i+1])<(data[[colname]][i+2]))
        | ((data[[colname]][i-2])<(data[[colname]][i-1]) & (data[[colname]][i-1])<(data[[colname]][i]) 
        & (data[[colname]][i])>(data[[colname]][i+1]) & (data[[colname]][i+1])>(data[[colname]][i+2]))
    ){
      maxvals <- c(maxvals, data[[colname]][i])
      indices <- c(indices, i)
    }

      
  }
  
  
  return(data[indices,])
  
}

ind<-c(0)
#getDerivatives move de vectors zoals in heave split en dan verschil
for (i in 1:length(relevant_sim_data$t))
{
  if (is.na(relevant_sim_data[i,"z_wf"])==TRUE)
  {
    ind <- c(ind, i)
  }
}
