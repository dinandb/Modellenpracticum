library(gridExtra)
library(ggplot2)
library(readxl)
library(dplyr)
# setwd("C:\\Users\\blomd\\Dropbox\\Documenten\\24-25 RU\\Modellenpracticum")

simulated_data <- read.csv("assets/M5415_10kn_JONSWAP_3m_10s/output.csv", header = TRUE, sep = ",")

# deze hierboven aanpassen naar nieuwe data

relevant_sim_data <- simulated_data[, c("t", "z_wf", "phi_wf", "theta_wf", "zeta")]
relevant_sim_data <- relevant_sim_data[-1, ] # remove the column name
relevant_sim_data <- data.frame(lapply(relevant_sim_data, as.numeric))  # Convert all columns to numeric
source("emma_dinand/logreg/QP_dinand.R")
source("emma_dinand/logreg/getMaxVals.R")
# source("emma_dinand/logreg/QP_emma.R")

relevant_sim_data$QP <- QP # voeg de QP (output variabele) toe
relevant_sim_data_backup <- relevant_sim_data
relevant_sim_data <- getMaxVals(relevant_sim_data_backup, "z_wf")






