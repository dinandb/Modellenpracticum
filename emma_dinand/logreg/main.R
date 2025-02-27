
# source("emma_dinand/logreg/prepare_clean_data.R")

source("emma_dinand/logreg/heave_split.R")
source("emma_dinand/logreg/divide_data.R")
source("emma_dinand/logreg/predicting QP's logres.R")

source("emma_dinand/logreg/errors.R")
source("emma_dinand/logreg/evaluate.R")

source("emma_dinand/logreg/getMaxVals.R")
# source("emma_dinand/logreg/logreg_for_predicting_QP's real data.R")

main <- function() {
  # we hebben nu relevant_sim_data klaar
  
  # nu evaluate
  
  find_min_confusion_matrix_error()
  
  
  
}

#table.count in error van matrix

#met threshodl spelen en kjiken of het klopt de fps en tps enzo
#smote adden