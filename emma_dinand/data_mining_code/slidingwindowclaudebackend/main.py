from data_engineering import init
from features import build_features
import LSTM_pytorch_train
from extras import split_data_scale, evaluate, display_information
import nonlinearsvm_train


def main():
    Xs, Ys = init(to_log=False)
    id = 3
    display_information(Xs[id], Ys[id])
    quit()
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_data_scale(Xs[id], Ys[id], id = id, new = False)
    # quit()
    # model = LSTM_pytorch_train.run(X_train_scaled, y_train, new = True)
    model = nonlinearsvm_train.train(X_train_scaled, y_train)

    evaluate(model, X_test_scaled, y_test)
    
    

main()