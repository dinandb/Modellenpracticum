# tuning hyperparameters

#absolute error
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
print(min_abs)
#best 0.3865511, 340, 2

#confusion matrix
min_conf <- c(confusion_matrix_error(6,3, 5, 1, 0.4), 6, 3, 5, 1, 0.4)
for(m in seq(200, 400, by = 40))
{
  for (n in 1:5)
  {
    for (o in seq(0.3, 0.9, by = 0.2))
    {
      new_error = confusion_matrix_error(m,n, 5, 1, o)
      if (new_error<min_conf[1])
      {
        min_conf <- c(new_error, m, n, 5, 1, o)
      }
    }

  }
}
print(min_conf)
#best 0.3496296 280, 1, 5, 1, 0.3


#accuracy
max_ac <- c(accuracy(6,1, 0.4), 6, 1, 0.4)
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


