
#rename deze naar iets van data mining
getMaxVals <- function(data, colname="z_wf") {
  # colname moet een string zijn
  
  maxvals <- c()
  
  new_data <- c()
  indices <- c()
  
  for (i in 2:(length(data[[colname]])-1)) {
    # if (data[[colname]][i-1] < data[[colname]][i] & data[[colname]][i] > data[[colname]][i+1]) {
    # if (abs(data[[colname]][i-1]) < abs(data[[colname]][i]) & abs(data[[colname]][i]) > abs(data[[colname]][i+1])) {
    if ((data[[colname]][i-1]) < (data[[colname]][i]) & (data[[colname]][i]) > (data[[colname]][i+1])
        | (data[[colname]][i-1]) > (data[[colname]][i]) & (data[[colname]][i]) < (data[[colname]][i+1])) {
      maxvals <- c(maxvals, data[[colname]][i])
      # new_data <- c(new_data, data[i,])
      indices <- c(indices, i)
      
      
    }
    # else if (data$QP[i]) {
    #   # print("added bc QP = 1")
    #   maxvals <- c(maxvals, data[[colname]][i])
    #   # new_data <- c(new_data, data[i,])
    #   indices <- c(indices, i)
    # }
  }
  return(data[indices,])
  
}

#getDerivatives move de vectors zoals in heave split en dan verschil

getDiffs <- function(data, colname)  {
  
}