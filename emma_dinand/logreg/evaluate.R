# tuning hyperparameters

#absolute error
find_min_absolute_error <- function() {
min_abs <- c(absolute_error(5,1), 5, 1)
for(k in 5:10)
{
  for (l in 1:5)
  {
    new_error = absolute_error(k,l)
    if (new_error<min_abs[1])
    {
      min_abs <- c(new_error, k, l)
    }
  }
}
# print(min_abs)
#best 0.3865511, 340, 2
}

#confusion matrix
find_min_confusion_matrix_error <- function() {

weight_FN <- 0.5
weight_FP <- 1.5
min_conf <- c(1000000000000000, 10, 1, weight_FP, weight_FN, 0.8)
min_error <- 0

for(amount_time_steps in c(1,2,3,4,5,6,7,8,9,10))
{
  print(amount_time_steps)
  for (step_size in 1:3)
  {
    # for (threshold in c(0.06906354-0.0005187227, 0.06906354-0.0005187227/2, 0.06906354, 0.06906354+0.0005187227/2, 0.06906354+0.0005187227))
    for (threshold in seq(0, 0.1, by=0.01))
    {
      # print(paste("Calling confusion_matrix_error with:", amount_time_steps, step_size, weight_FP, weight_FN, threshold))
      new_error <- confusion_matrix_error(amount_time_steps, step_size, weight_FP, weight_FN, threshold)
      # print(paste("Returned value:", new_error))

      if (new_error<min_conf[1])

        {
        # print("weigh_FP")
        # print(weight_FP)
        min_conf <- c(new_error, amount_time_steps, step_size, weight_FP, weight_FN, threshold)
      }
      # print(c(new_error, amount_time_steps, step_size, weight_FP, weight_FN, threshold))
    }

  }
}
print(min_conf)

#best 0.3496296 280, 1, 5, 1, 0.3
}


#accuracy
find_best_accuracy <- function() {
max_ac <- c(accuracy(1,1, 0.1), 6, 1, 0.4)
for(p in seq(200, 400, by = 20))
{
  for (q in 1:3)
  {
    for (r in seq(0.1, 0.5, by = 0.1))
    {
      new_ac = accuracy(p,q, r)
      if (new_ac>max_ac[1])
      {
        max_ac <- c(new_ac, p, q, r)
      }
    }
    
  }
}
print(max_ac)
#best 0.8196296 240, 3, 0.3
}

find_best_accuracy_SMOTE <- function() {
  max_ac <- c(accuracy())
}


