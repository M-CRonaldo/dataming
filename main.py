
import sklearn
import geohash
import math
import sys
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans


def task1_1(dataf):
    print("Info:")
    print(dataf.info())
    print("\nDescribe:")
    print(dataf.describe())
    print("\nHead:")
    print(dataf.head(10))
    print("\nTail:")
    print(dataf.tail(7))
    print("\n")


def task1_2(dataf):
    # 1.2
    dataf['pickup_geohash'] = dataf['pickup_geohash'].apply(geohash.decode)
    dataf['dropoff_geohash'] = dataf['dropoff_geohash'].apply(geohash.decode)
    # 1.2.1
    dataf['pickup_x'] = [float(i[0]) for i in dataf['pickup_geohash']]
    dataf['pickup_y'] = [float(i[1]) for i in dataf['pickup_geohash']]
    dataf['dropoff_x'] = [float(i[0]) for i in dataf['dropoff_geohash']]
    dataf['dropoff_y'] = [float(i[1]) for i in dataf['dropoff_geohash']]
    x1 = pd.array([i for i in dataf['pickup_x']])
    y1 = pd.array([i for i in dataf['pickup_y']])
    x2 = pd.array([i for i in dataf['dropoff_x']])
    y2 = pd.array([i for i in dataf['dropoff_y']])
    dataf['pickup_cor'] = [[x1[i], y1[i]] for i in range(len(x1))]
    dataf['dropoff_cor'] = [[x2[i], y2[i]] for i in range(len(x2))]


def task1_3(dataf):
    x1 = pd.array([i for i in dataf['pickup_x']])
    y1 = pd.array([i for i in dataf['pickup_y']])
    x2 = pd.array([i for i in dataf['dropoff_x']])
    y2 = pd.array([i for i in dataf['dropoff_y']])
    x = x2 - x1
    y = y2 - y1
    dis = [math.sqrt((x[i])**2 + (y[i])**2)
            for i in range(len(x))]
    dist = pd.array([i for i in dis])
    dataf['distance'] = dist
    #     lon1 = pd.array([math.radians(i) for i in dataf['pickup_x']])
    #     lat1 = pd.array([math.radians(i) for i in dataf['pickup_y']])
    #     lon2 = pd.array([math.radians(i) for i in dataf['dropoff_x']])
    #     lat2 = pd.array([math.radians(i) for i in dataf['dropoff_y']])
    #     dlon = lon1 - lon2   # 经度差
    #     dlat = lat1 - lat2   # 纬度差
    #     a = [math.sin(dlat[i]/2)**2 + math.cos(lat1[i]) * math.cos(lat2[i]) * math.sin(dlon[i]/2)**2
    #          for i in range(len(dlon))]
    #     c = pd.array([2 * math.asin(math.sqrt(i)) for i in a])
    # dataf['distance'] = c * 6371



def task1_4(dataf):
    # 1.4.1
    print(sys.getsizeof(dataf))
    print("Info:")
    print(dataf.info())
    # 1.4.2
    print(dataf.head(10))


def task1_5(dataf):
    prelen = len(dataf)
    dataf.dropna()
    dataf.drop(dataf[abs(dataf.pickup_x) > 90].index, inplace=True)
    dataf.drop(dataf[abs(dataf.pickup_y) > 90].index, inplace=True)
    dataf.drop(dataf[abs(dataf.dropoff_x) > 90].index, inplace=True)
    dataf.drop(dataf[abs(dataf.dropoff_y) > 90].index, inplace=True)
    dataf.drop(dataf[abs(dataf.distance) == 0].index, inplace=True)
    dataf.drop(dataf[dataf.fare <= 0].index, inplace=True)
    dataf.drop(dataf[dataf.passenger <= 0].index, inplace=True)
    print(dataf.describe())
    print("the number of rows removed is " + str(prelen - len(dataf)))
    return


def task1_6(dataf):
    print("Infomation about Int data: " + str(dataf['passenger'].describe()))
    print("\nEarliest Pickup Time: " + str(min(dataf['pickup_datetime'])))
    print("Latest Pickup Time: " + str(max(dataf['pickup_datetime'])))


def task1_7(dataf):
    copydf = dataf.copy(deep=False)
    copydf['pickup_datetime'] = copydf['pickup_datetime'].apply(pd.to_datetime)
    copydf.set_index(keys='pickup_datetime', inplace=True)
    print("Between 8 am and 9 am: " + str(len(copydf.between_time(start_time='08:00', end_time='09:00'))))
    print("Between 1 am and 2 am: " + str(len(copydf.between_time(start_time='01:00', end_time='02:00'))))


def task1(dataf):
    print("Task 1.1: -------------------------------------")
    task1_1(dataf)
    print("Task 1.2: -------------------------------------")
    task1_2(dataf)
    print("Task 1.3: -------------------------------------")
    task1_3(dataf)
    print("Task 1.4: -------------------------------------")
    task1_4(dataf)
    print("Task 1.5: -------------------------------------")
    task1_5(dataf)
    print("Task 1.6: -------------------------------------")
    task1_6(dataf)
    print("Task 1.7: -------------------------------------")
    task1_7(dataf)


def task2_1(dataf):
    # 2.1.1
    timelist, timeinter = [], []
    copydf = dataf.copy(deep=False)
    copydf['pickup_datetime'] = copydf['pickup_datetime'].apply(pd.to_datetime)
    copydf.set_index(keys='pickup_datetime', inplace=True)
    for i in range(96):   # 24*4=96
        timelist.append(str(int(i * 15 / 60)) + ":" + str(i * 15 % 60))
        if timelist[-1][-2:] == ":0":
            timelist[-1] = timelist[-1] + "0"  # 时间补0
    for i in range(len(timelist)-1):
        timeinter.append(len(copydf.between_time(start_time=timelist[i], end_time=timelist[i+1])))
    timeinter.append(len(copydf.between_time(start_time=timelist[-1], end_time=timelist[0])))  # 23:45--0:00
    # 2.1.2
    rearray = pd.DataFrame()
    rearray['timelist'] = timelist
    rearray['number'] = timeinter
    # rearray.set_index(keys='timelist', inplace=True)
    print(rearray)
    return rearray


def task2_2(dataf):
    X_cor = []
    Y_cor = []
    for i in list(dataf['pickup_cor']):
        X_cor.append(i)
    for i in list(dataf['dropoff_cor']):
        Y_cor.append(i)
    X = X_cor + Y_cor
    estimator = KMeans(n_clusters=30)
    estimator.fit(X)
    # print(X)
    pickuppred = estimator.predict(X_cor)
    dropoffpred = estimator.predict(Y_cor)
    dataf['pickup_cluster'] = pickuppred
    dataf['dropoff_cluster'] = dropoffpred
    print("The percentage of orders has started from a cluster centers and ends at the same cluster centers: ")
    print("\t" + str(len(dataf.loc[dataf['pickup_cluster'] == dataf['dropoff_cluster']])/len(dataf)*100) + "%")
    return estimator.cluster_centers_
    # 2.2.1
    # clusterx = list(dataf['dropoff_x'])
    # for i in list(dataf['pickup_x']):
    #     clusterx.append(i)
    # clustery = list(dataf['dropoff_y'])
    # for i in list(dataf['pickup_y']):
    #     clustery.append(i)
    # pos = np.mat([clusterx, clustery])
    # estimator = KMeans(n_clusters=30).fit(pos.T)
    # 2.2.2
    #     # pickuppos = np.array(np.mat([list(dataf['pickup_x']), list(dataf['pickup_y'])]).T)
    #     # dropoffpos = np.array(np.mat([list(dataf['dropoff_x']), list(dataf['dropoff_y'])]).T)
    #     # pickuppred = estimator.predict(pickuppos)
    #     # dropoffpred = estimator.predict(dropoffpos)



def task2(dataf):
    print("Task 2.1: -------------------------------------")
    re1 = task2_1(dataf)
    print("Task 2.2: -------------------------------------")
    re2 = task2_2(dataf)
    return re1, re2


def task3_1(taskdata):
    taskdata.plot(y='number', x='timelist', kind='line')
    plt.pyplot.show()


def task3_2(dataf):
    countlist = np.mat(dataf.groupby('pickup_cluster').count()).T.tolist()[0]
    for i in range(len(countlist)):
        countlist[i] += np.mat(dataf.groupby('dropoff_cluster').count()).T.tolist()[0][i]
    tmparray = pd.DataFrame()
    tmparray['number'] = countlist
    tmparray.plot(y='number', kind='bar')
    plt.pyplot.show()


def task3_3(dataf, taskdata):
    tmpx = np.mat(taskdata).T.tolist()[0]
    tmpy = np.mat(taskdata).T.tolist()[1]
    tmpdf2 = pd.DataFrame()
    tmpdf2['center_x'] = tmpx
    tmpdf2['center_y'] = tmpy
    tmpdf1 = dataf.sample(n=100)
    tmpdf1.plot.scatter(x='pickup_x', y='pickup_y', c='blue')
    tmpdf1.plot.scatter(x='dropoff_x', y='dropoff_y', c='red')
    tmpdf2.plot.scatter(x='center_x', y='center_y', c='black')
    plt.pyplot.show()


def task3(dataf, task3_1data, task3_3data):
    task3_1(task3_1data)
    task3_2(dataf)
    task3_3(dataf, task3_3data)


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('taxi_train.csv')
    task1(df)
    task3_1d, task3_3d = task2(df)
    task3(df, task3_1d, task3_3d)
