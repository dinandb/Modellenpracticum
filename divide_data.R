


split <- function(data) {
  n = 18001
  train_data <- relevant_sim_data[1:floor(0.7 * n),]
  validation_data <- relevant_sim_data[floor(0.7*n+1):floor(0.85*n),]
  test_data <- relevant_sim_data[floor(0.85*n+1):n,]
  # train_data <- QP[1:floor(0.7 * n), ]  # Take the first 70% of the rows
  print("trian data")
  print(head(test_data))
  return (list(train_data=train_data, val=validation_data, test=test_data))
}


split_data <- function(data) {
  n = 18001
  train_data <- data[1:floor(0.7 * n),]
  validation_data <- data[floor(0.7*n+1):floor(0.85*n),]
  test_data <- data[floor(0.85*n+1):n,]
  # train_data <- QP[1:floor(0.7 * n), ]  # Take the first 70% of the rows
  return (list(train_data=train_data, val=validation_data, test=test_data))
}
