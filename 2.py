import datetime

import xgboost
import numpy as np

import xgboost_predictor


if __name__ == "__main__":
    data = []
    with open("tests/resources/data/agaricus.txt.0.test", "r") as f:
        for line in f.readlines():
            row = [0] * 126
            for i in line.split(" ")[1:]:
                f, v = i.split(":")
                row[int(f)] = int(v)
            data.append(row)

    booster1 = xgboost.Booster({"nthread": 1})
    booster1.load_model('tests/resources/model/gblinear/v40/binary-logistic.model')
    data1 = xgboost.DMatrix(np.array(data))
    
    start = datetime.datetime.now()
    a = booster1.predict(data1)
    print(datetime.datetime.now() - start)
    
    booster2 = xgboost_predictor.load_model('tests/resources/model/gblinear/v40/binary-logistic.model')
    data2 = [[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]]

    start = datetime.datetime.now()
    b = booster2.predict_batch(data)
    print((datetime.datetime.now() - start))

    # print(a, b)
