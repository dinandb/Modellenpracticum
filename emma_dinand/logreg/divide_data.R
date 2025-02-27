


split_data <- function(data) {
  n = length(data$t)
  train_data <- data[1:floor(0.7 * n),]
  validation_data <- data[floor(0.7*n+1):floor(0.85*n),]
  test_data <- data[floor(0.85*n+1):n,]
  # train_data <- QP[1:floor(0.7 * n), ]  # Take the first 70% of the rows
  return (list(train_data=train_data, val=validation_data, test=test_data))
}
