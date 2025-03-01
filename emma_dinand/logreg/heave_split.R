




generate_prev_heavs <- function(steps, time_steps, data) {
  n = length(data$t)

  relevant_sim_data_splitted <- data
  #print(relevant_sim_data_splitted)
  for (i in steps:1) {
    col_name <- paste0("heave", i)  # Create column name dynamically
    #print(col_name)
    relevant_sim_data_splitted[[col_name]] <- relevant_sim_data_splitted$z_wf
    
    relevant_sim_data_splitted[[col_name]] <- dplyr::lag(relevant_sim_data_splitted$z_wf, n = time_steps*i, default = NA)  
    
    
  }
  return(relevant_sim_data_splitted)
  
}








