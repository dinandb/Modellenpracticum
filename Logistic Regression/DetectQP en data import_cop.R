 library(gridExtra)
 library(ggplot2)
 library(readxl)
 library(dplyr)
 
 setwd("C:/Users/exemm/OneDrive/Documenten/3e jaar Radboud Bachelor/Modellen practicum github/Modellenpracticum")
 
 simulated_data <- read.csv("assets/finaldata2.csv", header = TRUE, sep = ",")
 relevant_sim_data <- simulated_data
 relevant_sim_data <- relevant_sim_data[-1, ] # remove the column name
 relevant_sim_data <- data.frame(lapply(relevant_sim_data, as.numeric))  # Convert all columns to numeric


data <- relevant_sim_data #in logreg_for_predicting_QP's.R
QP<-c(0)
i<-2
while (i <=length(data$t))
{
  if (data$QP1[i] == 1)
  {
    h <-0
    while (data$QP1[h+i] == 1)
    {
      h <- h + 1
    }
    if (h>149)
    {
      QP <- append(QP, rep(TRUE,h-149))
      QP <- append(QP, rep(FALSE,149))
    }
    else
    {
      QP<- append (QP, rep(FALSE, h))
    }
    i <- i + h
  }
  else
  {
    QP <- append (QP, FALSE)
    i <- i+1
  }
}
data$QP<-QP
relevant_sim_data <- data

relevant_sim_data_backup <- relevant_sim_data



relevant_sim_data$QP <- QP # voeg de QP (output variabele) toe


