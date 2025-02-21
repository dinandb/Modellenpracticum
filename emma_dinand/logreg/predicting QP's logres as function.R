logres <- function(steps,time_steps) {
  # library(dplyr)
  # setwd("C:/Users/exemm/OneDrive/Documenten/3e jaar Radboud Bachelor/Modellen practicum")
  
  

  
  # maak ze numeric
  relevant_sim_data$QP <- QP # voeg de QP (output variabele) toe
  # source("heave_split.R")
  
  relevant_sim_data_final <- generate_prev_heavs(steps, time_steps, relevant_sim_data)
  
  
  # source("divide_data.R")
  
  
  # dit is alle data, met alle nieuwe heave columns
  
  
  relevant_sim_data_final_splitted <- split_data(relevant_sim_data_final)
  relevant_sim_data_final_train <- relevant_sim_data_final_splitted$train_data
  relevant_sim_data_final_eval <- relevant_sim_data_final_splitted$val
  relevant_sim_data_final_test <- relevant_sim_data_final_splitted$test
  # 
  # source("voor emma/DetectQP.R")
  
  
  # Generate the heave variable names dynamically (heave1, heave2, ..., heaveN)
  heave_vars <- paste0("heave", 1:steps)
  
  # Create a formula for the glm model dynamically
  formula_str <- paste("QP ~", paste(heave_vars, collapse = " + "), "+ z_wf")
  
  # Convert the formula string to an actual formula object
  formula <- as.formula(formula_str)
  
  last_heave <- paste0("heave", steps)
  
  logreg_model <- glm(formula, data = relevant_sim_data_final_train[!is.na(relevant_sim_data_final_train[[last_heave]]),], family = binomial)
  
  #evaluation data into trained model
  new_data = relevant_sim_data_final_eval[is.na(relevant_sim_data_final_eval[[last_heave]]) != TRUE,]
  
  new_data$Probability <- predict(logreg_model, newdata = new_data, type = "response")
  
  

  
  
  return(new_data)
  
}

