library(gridExtra)
library(ggplot2)
library(readxl)
library(dplyr)
# setwd("C:/Users/exemm/OneDrive/Documenten/3e jaar Radboud Bachelor/Modellen practicum")
# 
# simulated_data <- read.csv("M5415_10kn_JONSWAP_3m_10s/output.csv", header = TRUE, sep = ",")
relevant_sim_data <- simulated_data[, c("t", "z_wf", "phi_wf", "theta_wf", "zeta")]
relevant_sim_data <- relevant_sim_data[-1, ] # remove the column name
relevant_sim_data <- data.frame(lapply(relevant_sim_data, as.numeric))  # Convert all columns to numeric


Data <- relevant_sim_data #in logreg_for_predicting_QP's.R
#Data <- data.frame(lapply(Data, as.numeric))  # Convert all columns to numeric
time <- (Data$t)
heave <- (Data$z_wf)
roll <- (Data$phi_wf)

heaveThres <- 0.2
rollThres <- 0.02
timeThres <- 30

QP <- c(0)
QPstart <- c(0)
QPend <- c(0)



i <- 1
while(i < length(time)) 
{
  if(abs(heave[i]) < heaveThres #&& roll[i] < rollThres
     ) {
    j <- 0
    while(i+j < length(time) & abs(heave[i + j]) < heaveThres #&& roll[i + j] < rollThres && i + j < length(time)
          )
    {
      j <- j + 1
    }
    if(time[i + j] - time[i] >= timeThres)
    {
      QP <- append(QP, rep(TRUE,j-timeThres +1))
      QP <- append(QP, rep(FALSE,timeThres -1))
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


