absolute_error <- function(stps, t_stps) {
  
  new_data <- logres(stps, t_stps)
  
  error = 0
  for (i in 1:length(new_data$Probability)) {
    error = error + abs(new_data$Probability[i]-new_data$QP[i])
  }
  
  return (error/length(new_data$Probability))
  
}

confusion_matrix_error <- function(stps, t_stps, false_pos, false_neg, bar) {
  
  new_data <- logres(stps, t_stps)
 
  error = 0
  for (i in 1:length(new_data$Probability)) {
    if (new_data$Probability[i]>=bar && new_data$QP[i] ==0)
    {
      error = error + false_pos}
    if (new_data$Probability[i]<bar && new_data$QP[i] ==1)
    {
      error = error + false_neg}
    
  }
  
  return (error/length(new_data$Probability))
  
}


accuracy <- function(stps, t_stps, bar) {
  
  new_data <- logres(stps, t_stps)
  
  count = 0
  for (i in 1:length(new_data$Probability)) {
    if (new_data$Probability[i]>=bar && new_data$QP[i] ==1)
    {
      count = count +1
    }
    if (new_data$Probability[i]<bar && new_data$QP[i] ==0)
    {
      count = count +1}
    
    }
  
  return (count/length(new_data$Probability))
  
}

