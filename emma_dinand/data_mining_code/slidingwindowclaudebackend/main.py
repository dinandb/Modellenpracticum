from data_engineering import init
from features import build_features
import LSTM_pytorch_train
from extras import split_data_scale, evaluate, display_information
import nonlinearsvm_train
from random_forest_QP_predict import model_train

def main():
    Xs, Ys = init(to_log=False)
    id = 2
    print(f'number QP = {sum(Ys[id])}, len = {len(Ys[id])}')
    # display_information(Xs[id], Ys[id])
    # quit()
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_data_scale(Xs[id], Ys[id], id = id, new = True)
    # quit()
    print(f"len X_train_scaled = {len(X_train_scaled)}")
    print(f"len X_test_scaled = {len(X_test_scaled)}")
    # model = LSTM_pytorch_train.run(X_train_scaled, y_train, new = True)
    model = nonlinearsvm_train.train(X_train_scaled, y_train)


    # model, scaler, X_test, y_test = model_train(Xs[id], Ys[id], new = True)
    # X_test_scaled = scaler.transform(X_test)
    evaluate(model, X_test_scaled, y_test)
    
    

main()