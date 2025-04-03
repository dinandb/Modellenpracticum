from datetime import datetime
from random_forest_QP_predict import *
from extras import save_processed_data, load_processed_data
import Detect_QP_CasperSteven

scaler = None
model = None
data = None
last_three_extrema = [None, None, None]
threshold = 0.5
counts_to_QP = []
total_predicted_true = 0

def web_app_driver():
    pass


def init(data_id=3, model_id=3):
    global counts_to_QP
    def make_counts_to_QP(y):
        counts_to_QP = []
        to_break = False
        i = 1
        while not to_break:
            elt = y[-i]
            if not elt:
                counts_to_QP.append(999999999999)
                i += 1
            else:
                to_break = True
        cur_away_from_QP = 0
        while i <= len(y):
            if y[-i]:
                counts_to_QP.append(0)
                cur_away_from_QP = 0
            else:
                cur_away_from_QP += 1
                counts_to_QP.append(cur_away_from_QP)
            i += 1
        counts_to_QP.reverse()
        return counts_to_QP
    
    
    global model, data, scaler

    data_file_path = f"slidingwindowclaudebackend/pickle_saves/data/processed_data_clean{str(data_id)}.pkl"
    model_file_path = f"slidingwindowclaudebackend/pickle_saves/modellen/model{str(model_id)}.pkl"
    try:
        data = load_processed_data(data_file_path)
        print(f"Data loaded from {data_file_path}")
    except FileNotFoundError:
        print(f"Data file {data_file_path} not found. Loading default data.")
        file_path = f'../../assets/data{data_id}.csv'
        data = pd.read_csv(file_path, header=[0,1])
        data = data[['t', 'z_wf', 'Delta_t', 'y_wf', 'x_wf', 'psi_wf', 'phi_wf', 'theta_wf']]
        
        data = data.apply(pd.to_numeric, errors='coerce')
        save_processed_data(data, data_file_path)
        print("Processed data saved to pickle.")


    try:
        model, scaler, _, _ = load_processed_data(model_file_path)
        print(f"Data loaded from {model_file_path}")
    except FileNotFoundError:
        print(f"Model file {model_file_path} not found. training modl.")
        
        
        model, scaler, _, _ = model_train()

        save_processed_data(model, model_file_path)
        print("model data saved to pickle.")
    

    # maak die ene vector # until QP given entire dataset (counts_to_QP)
    y = Detect_QP_CasperSteven.mark_QP(data,name=f"QP{model_id}", new=False)
    counts_to_QP = make_counts_to_QP(y)

def get_data_point():
    global model, data, last_three_extrema, total_predicted_true
    
    
    # go through the data, when we have three extrema we can start predicting.
    # every datapoint, check if it is an extrema
    i = 0
    
    while i < len(data):
        # print(i)
        is_safe = False
        # Check if the current point is an extremum
        if i >= 4:
            # check if the data.iloc[i-2] is extremum

                # Check if the current point is an extremum using absolute values
            if abs(data.iloc[i-2]['z_wf']) > abs(data.iloc[i-3]['z_wf']) > abs(data.iloc[i-4]['z_wf']) and \
                abs(data.iloc[i-2]['z_wf']) > abs(data.iloc[i-1]['z_wf']) > abs(data.iloc[i]['z_wf']):
                # extremum found
                
                
                # Update the last_three_extrema list



                # TODO checken of de laatste extremum aan t begin of einde is (wat is de volgorde?)
                # ans: volgorde is oud naar nieuw
                
                last_three_extrema = [last_three_extrema[1], last_three_extrema[2], data.iloc[i-2]['z_wf']] #checked
                print(last_three_extrema)
            # now, check if all extrema are not None, then we can predict
            # if one of the extrema is None, just return false


            # TODO checken wat precies de format was waar het model op getraind is
            if last_three_extrema[0] is not None and last_three_extrema[1] is not None and last_three_extrema[2] is not None:
                # we have three extrema, we can predict


                X_scaled = scaler.transform([last_three_extrema])
    
                prediction = model.predict_proba(X_scaled)
                # TODO shape van prediction checken

                bin_prediction = (prediction[:, 1] > threshold)[0]
                
                if bin_prediction == 1:
                    is_safe = True
                    total_predicted_true += 1
                else:
                    is_safe = False
        
        # TODO format: {"name": datetime.now().strftime("%H:%M:%S"), "value": heave_waard, "timestamp": datetime.now().isoformat(), "is_safe": Bool, "amount_to_go_for_safe": Int}
        value = data.iloc[i]['z_wf']
        # yield {"value": round(value, 2), "is_safe": is_safe, "amount_to_go_for_safe": counts_to_QP[i]}
        i += 1
        if i % 100 == 0:
            print(i)
            print(f"total predicted true {total_predicted_true}")


# deze functies moeten gecalled kunnen worden mbv een button in de webapp
def change_threshold(new_threshold):
    global threshold
    threshold = new_threshold
    print(f"Threshold changed to {threshold}")

def change_model(new_model_id):
    global model
    model_file_path = f"slidingwindowclaudebackend/pickle_saves/modellen/model{str(new_model_id)}.pkl"
    try:
        model = load_processed_data(model_file_path)
        print(f"Model loaded from {model_file_path}")
    except FileNotFoundError:
        print(f"Model file {model_file_path} not found.")

init()
data_generator = get_data_point()
# for i in range(5000):
#     if i % 100 == 0:
#         print(i)
#         print(f"total predicted true {total_predicted_true}")
#     (next(data_generator))
    