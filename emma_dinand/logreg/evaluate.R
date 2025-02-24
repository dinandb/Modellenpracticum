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

weight_FN <- 1
weight_FP <- 5
min_conf <- c(confusion_matrix_error(1,1, weight_FP, weight_FN, 0.4), 1, 1, weight_FP, weight_FN, 0.4)
for(amount_time_steps in seq(4,4, by = 1))
{
  print(amount_time_steps)
  for (step_size in 4:4)
  {
    for (threshold in seq(0.15, 0.25, by = 0.01))
    {
      new_error = confusion_matrix_error(amount_time_steps, step_size, 5, 1, threshold)
      if (new_error<min_conf[1])
      {
        min_conf <- c(new_error, amount_time_steps, step_size, weight_FP, weight_FN, threshold)
      }
      print(c(new_error, amount_time_steps, step_size, weight_FP, weight_FN, threshold))
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


