#getDerivatives move de vectors zoals in heave split en dan verschil

getDiffs <- function(data, colname="z_wf", steps=1, power=1)  {
  
  n = length(data$t)
  
  relevant_sim_data_derivatives <- data
  
  
  der0 <- c(relevant_sim_data_derivatives[1:n,colname])
  der1<-c(0)
  
  for(j in 1: power)
  {
    for (k in 2:n)
    {
      der1 [k] <- der0[k] - der0[k-1]
    }
    der0 <- der1
  }
  
  der0[1]<- NA

  for (i in steps:1) {
    col_name <- paste0("Derivative", i)  # Create column name dynamically

    #relevant_sim_data_derivatives[[col_name]] <- der0 
    
    relevant_sim_data_derivatives[[col_name]] <- dplyr::lag(der0, n = i-1, default = NA)  
    
  }

  return(relevant_sim_data_derivatives)
}

getDiffs2 <- function(data, colname="z_wf", highest_p=1)  {
  
  n = length(data$t)
  
  relevant_sim_data_derivatives <- data
  
  
  der0 <- c(relevant_sim_data_derivatives[1:n,colname])
  der1<-c(0)
  
  for(j in 1: highest_p)
  {
    col_name <- paste0("Derivative", j)
    for (k in 2:n)
    {
      der1 [k] <- der0[k] - der0[k-1]
    }
    der0 <- der1
    relevant_sim_data_derivatives[[col_name]] <- der1
  }
  
  der0[1]<- NA
  
  
  return(relevant_sim_data_derivatives)
}

