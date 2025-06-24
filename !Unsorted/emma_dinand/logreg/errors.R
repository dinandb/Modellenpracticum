absolute_error <- function(stps, t_stp) {
  
  predicted_probs_QPs <- logreg_predict_eval(stps, t_stp)
  predicted_probs = predicted_probs_QPs$probs
  QP_eval = predicted_probs_QPs$QPs
  # hier moeten de probs en de echte qps nog uitgehaald worden
  
  error = 0
  for (i in 1:length(predicted_probs)) {
    error = error + abs(predicted_probs[[i]]-QP_eval[i])
  }
  
  return (error/length(predicted_probs))
  
}

confusion_matrix_error <- function(stps=1, t_stp=1, false_pos_weight=1.8, false_neg_weight=0.2, bar=0.05) {
  
  predicted_probs_QPs <- logreg_predict_eval(stps, t_stp)
  predicted_probs = predicted_probs_QPs$probs
  QP_eval = predicted_probs_QPs$QPs
  # bar = mean(predicted_probs) + 3*sd(predicted_probs)
  
  print(mean(predicted_probs))
  print(2*sd(predicted_probs))
  
  # indices <- seq(1, length(predicted_probs))
  FP <- 0
  FN <- 0
  for (i in (1:length(predicted_probs))) {
    if (predicted_probs[[i]] >= bar & QP_eval[i] == 0) {
      FP <- FP + 1
    }
    if (predicted_probs[[i]] < bar & QP_eval[i] == 1) {
      FN <- FN + 1
    }
  }
  # false_pos_indices <- (predicted_probs >= bar) & (predicted_probs == 0)
  # false_neg_indices <- (predicted_probs < bar) & (predicted_probs == 1)
  # FP = sum(false_pos_indices)
  # FN = sum(false_neg_indices)
  # print("IPV DE GETALLEN HIERONDER IN CONFUSION MATRIX ERROR IETS VAN TABLE.COUNT")
  TP = length(QP_eval[QP_eval==1]) - FN
  TN = length(QP_eval[QP_eval==0]) - FP
  print("FP,FN,TP,TN")
  print(FP)
  print(FN)
  print(TP)
  print(TN)
  
  error = FP*false_pos_weight + FN * false_neg_weight
  print("error")
  print(error)
  confusion_matrix <- matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE)
  colnames(confusion_matrix) <- c("Predicted 1", "Predicted 0")
  rownames(confusion_matrix) <- c("Actual 1", "Actual 0")
  
  
  # print(confusion_matrix)
  
  
  return (error)
  
}


accuracy <- function(stps, t_stp, bar) {
  
  predicted_probs_QPs <- logreg_predict_eval(stps, t_stp)
  predicted_probs = predicted_probs_QPs$probs
  # print(predicted_probs)
  QP_eval = predicted_probs_QPs$QPs
  # print(QP_eval)
  
  count = 0
    if (predicted_probs[[i]]>=bar && QP_eval[i] ==1)
    {
      count = count +1
    }
    if (predicted_probs[[i]]<bar && QP_eval[i] ==0)
    {
      count = count +1
    
    }
  
  return (count/length(predicted_probs))
  
}



# confusion_matrix_error_diff <- function(stps=1, power=1, false_pos_weight=1.8, false_neg_weight=0.2, bar=0.5) {
#   
#   predicted_probs_QPs <- logreg_predict_eval_diff(stps, power)
#   predicted_probs = predicted_probs_QPs$probs
#   QP_eval = predicted_probs_QPs$QPs
#   
#   bar = mean(predicted_probs) + 2*sd(predicted_probs)
#   
#   # indices <- seq(1, length(predicted_probs))
#   FP <- 0
#   FN <- 0
#   for (i in (1:length(predicted_probs))) {
#     if (predicted_probs[[i]] >= bar & QP_eval[i] == 0) {
#       FP <- FP + 1
#     }
#     if (predicted_probs[[i]] < bar & QP_eval[i] == 1) {
#       FN <- FN + 1
#     }
#   }
#   # false_pos_indices <- (predicted_probs >= bar) & (predicted_probs == 0)
#   # false_neg_indices <- (predicted_probs < bar) & (predicted_probs == 1)
#   # FP = sum(false_pos_indices)
#   # FN = sum(false_neg_indices)
#   # print("IPV DE GETALLEN HIERONDER IN CONFUSION MATRIX ERROR IETS VAN TABLE.COUNT")
#   TP = length(QP_eval[QP_eval==1]) - FN
#   TN = length(QP_eval[QP_eval==0]) - FP
# 
#   
#   error = FP*false_pos_weight + FN * false_neg_weight
#   
#   confusion_matrix <- matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE)
#   colnames(confusion_matrix) <- c("Predicted 1", "Predicted 0")
#   rownames(confusion_matrix) <- c("Actual 1", "Actual 0")
#   
#   
#   print(confusion_matrix)
#   
#   
#   return (error)
#   
# }
