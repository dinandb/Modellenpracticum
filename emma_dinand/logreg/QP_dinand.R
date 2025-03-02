# library(gridExtra)
# library(ggplot2)
# library(readxl)
# library(dplyr)
# setwd("C:/Users/exemm/OneDrive/Documenten/3e jaar Radboud Bachelor/Modellen practicum")
# 
# simulated_data <- read.csv("M5415_10kn_JONSWAP_3m_10s/output.csv", header = TRUE, sep = ",")
# relevant_sim_data <- simulated_data[, c("t", "z_wf", "phi_wf", "theta_wf", "zeta")]
# relevant_sim_data <- relevant_sim_data[-1, ] # remove the column name
# relevant_sim_data <- data.frame(lapply(relevant_sim_data, as.numeric))  # Convert all columns to numeric


Data <- relevant_sim_data_backup #in logreg_for_predicting_QP's.R
#Data <- data.frame(lapply(Data, as.numeric))  # Convert all columns to numeric
time <- (Data$t)
heave <- (Data$z_wf)
roll <- (Data$phi_wf)

heaveThres <- 0.5
rollThres <- 0.02
timeThres <- 30

QP <- c(0)
QPstart <- c(0)
QPend <- c(0)

num_start_QP <- 2


i <- 1
while(i < length(time)) 
{
  if(heave[i] < heaveThres) {  #} && roll[i] < rollThres) {
    j <- 0
    # print(i+j)
    while((i+j) < length(heave) & heave[i + j] < heaveThres)# && roll[i + j] < rollThres && i + j < length(time))
    {
      j <- j + 1
    }
    if(time[i + j] - time[i] >= timeThres)
    {
      QP <- append(QP, rep(TRUE,num_start_QP))
      QP <- append(QP, rep(FALSE,j-num_start_QP))
      QPstart <- append(QPstart, time[i])
      QPend <- append(QPend, time[i+j])
    }
    else if(time[i + j] - time[i] < timeThres)
    {
      QP <- append(QP, rep(FALSE,j))
    }
    i <- i+j
  }
  else
  {
    QP <- append(QP,FALSE)
    i <- i + 1
  }
}

moveQP <- function(QP, amountToMove = 3) {
  amountToAdd <- amountToMove - num_start_QP
  
  # remove first 5 of the vector, add 5 zeros to the end.
  
  # then go through the vector, each last 1 we have, change the next amountToAdd 0's to 1
  
  QP <- QP[(amountToMove+1):length(QP)]
  QP <- c(QP, rep(0,amountToMove))
  toSkip = 0
  for (i in 1:(length(QP)-1)) {
    if (QP[i] == 1 & QP[i+1] == 0 & toSkip <= 0) {
      for (j in 1:amountToAdd) {
        QP[i+j] = 1
      }
      toSkip = amountToAdd
    }
    else {
      toSkip = toSkip - 1
    }
  }
  return(QP)
}


