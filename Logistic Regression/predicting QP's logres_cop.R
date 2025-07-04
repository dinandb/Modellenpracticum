

logreg_predict_eval <- function(steps,time_step) {
  
  relevant_sim_data_final <- generate_prev_heavs(steps, time_step, relevant_sim_data)

  relevant_sim_data_final_splitted <- split_data(relevant_sim_data_final)
  relevant_sim_data_final_train <- relevant_sim_data_final_splitted$train_data
  relevant_sim_data_final_eval <- relevant_sim_data_final_splitted$val
  relevant_sim_data_final_test <- relevant_sim_data_final_splitted$test
  
  # Generate the heave variable names dynamically (heave1, heave2, ..., heaveN)
  heave_vars <- paste0("heave", 1:steps)
  
  # Create a formula for the glm model dynamically
  formula_str <- paste("QP ~", paste(heave_vars, collapse = " + "), "+ z_wf")
  # formula_str <- paste("QP ~", paste(heave_vars, collapse = " + "))
  
  # Convert the formula string to an actual formula object
  formula <- as.formula(formula_str)
  
  last_heave <- paste0("heave", steps)
  # print(formula)
  
  logreg_model <- glm(formula, data = relevant_sim_data_final_train[!is.na(relevant_sim_data_final_train[[last_heave]]),], family = binomial)
  print(summary(logreg_model))
  #evaluation data into trained model
  new_data = relevant_sim_data_final_eval[is.na(relevant_sim_data_final_eval[[last_heave]]) != TRUE,]
  
 
  probs <- predict(logreg_model, newdata = new_data, type = "response")

  
  return(list(probs=probs, QPs=new_data$QP))
  
}


