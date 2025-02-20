# simulated_data <- read.csv("assets/M5415_10kn_JONSWAP_3m_10s/output.csv", header = TRUE, sep = ",")
# relevant_sim_data <- simulated_data[, c("t", "z_wf", "phi_wf", "theta_wf", "zeta")]
# relevant_sim_data <- relevant_sim_data[-1, ] # remove the column name
# relevant_sim_data <- data.frame(lapply(relevant_sim_data, as.numeric))  # Convert all columns to numeric
# 

source("emma_dinand/logreg/predicting QP's logres as function.R")
source("emma_dinand/logreg/divide_data.R")
source("emma_dinand/logreg/errors.R")
source("emma_dinand/logreg/heave_split.R")
source("emma_dinand/logreg/evaluate.R")
source("emma_dinand/logreg/logreg_for_predicting_QP's real data.R")

