



gen_err <- function(stps=3, t_stp=1, false_pos_weight=1.8, false_neg_weight=0.2, bar=0.5) {
  
  relevant_sim_data_final <- generate_prev_heavs(stps, t_stp, relevant_sim_data)
  
  relevant_sim_data_final_splitted <- split_data(relevant_sim_data_final)
  relevant_sim_data_final_train <- relevant_sim_data_final_splitted$train_data
  relevant_sim_data_final_eval <- relevant_sim_data_final_splitted$val
  relevant_sim_data_final_test <- relevant_sim_data_final_splitted$test
  
  heave_vars <- paste0("heave", 1:stps)
  
  # Create a formula for the glm model dynamically
  formula_str <- paste("QP ~", paste(heave_vars, collapse = " + "), "+ z_wf")
  # formula_str <- paste("QP ~", paste(heave_vars, collapse = " + "))
  
  # Convert the formula string to an actual formula object
  formula <- as.formula(formula_str)
  
  last_heave <- paste0("heave", stps)
  
  logreg_model <- glm(formula, data = relevant_sim_data_final_train[!is.na(relevant_sim_data_final_train[[last_heave]]),], family = binomial)
  
  
  new_data = relevant_sim_data_final_test[is.na(relevant_sim_data_final_test[[last_heave]]) != TRUE,]
  
  
  
  predicted_probs = predict(logreg_model, newdata = new_data, type = "response")
  
  QP_test = new_data$QP
  
  # indices <- seq(1, length(predicted_probs))
  FP <- 0
  FN <- 0
  for (i in (1:length(predicted_probs))) {
    if (predicted_probs[[i]] >= bar & QP_test[i] == 0) {
      FP <- FP + 1
      print("FP!")
      # print(new_data[i,])
    }
    if (predicted_probs[[i]] < bar & QP_test[i] == 1) {
      FN <- FN + 1
    }
    if (predicted_probs[[i]] >= bar & QP_test[i] == 1) {
      
      print("TP!")
      # print(new_data[i,])
    }
  }
  
  TP = length(QP_test[QP_test==1]) - FN
  TN = length(QP_test[QP_test==0]) - FP
  
  error = FP*false_pos_weight + FN * false_neg_weight
  
  confusion_matrix <- matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE)
  colnames(confusion_matrix) <- c("Predicted 1", "Predicted 0")
  rownames(confusion_matrix) <- c("Actual 1", "Actual 0")
  
  
  print(confusion_matrix)
  
  
}
