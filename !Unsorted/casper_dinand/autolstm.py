from neuralforecast.auto import AutoLSTM, AutoNHITS
import dataimport


def main():
    print(neuralforecast.__version__)
    datas, ys = dataimport.init()
    data4 = datas[3]
    y4 = ys[3]


    



if __name__ == "__main__":
    main()


