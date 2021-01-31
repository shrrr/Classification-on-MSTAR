import numpy as np
from PIL import Image
from sklearn.svm import SVC
import os
from sklearn.decomposition import PCA
import imageio
from sklearn.metrics import classification_report

def get_mstar_data(stage, width=100, height=100, crop_size=100, aug=False):
    data_dir = "C:/Users/Eric/Desktop/Target-Detection-in-MSTAR-Images-master/train/" if stage == "train" else "C:/Users/Eric/Desktop/Target-Detection-in-MSTAR-Images-master/test/" if stage == "test" else None
    print("------ " + stage + " ------")
    sub_dir = ["2S1", "SN_132", "BRDM_2", "BTR_60", "SN_9563", "D7", "T62", "SN_C71", "ZIL131", "ZSU_23_4"]
    X = []
    y = []

    for i in range(len(sub_dir)):
        tmp_dir = data_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if (x.endswith(".jpg")or x.endswith(".JPG"))]
        print(sub_dir[i], len(img_idx))
        y += [i] * len(img_idx)
        for j in range(len(img_idx)):
            img = np.array(Image.fromarray(imageio.imread((tmp_dir + img_idx[j]))).resize((height, width)))
            img = img[(height - crop_size) // 2 : height - (height - crop_size) // 2, \
                  (width - crop_size) // 2: width - (width - crop_size) // 2]
            X.append(img)

    return np.asarray(X), np.asarray(y)

def data_shuffle(X, y, seed=0):
    data = np.hstack([X, y[:, np.newaxis]])
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]

# 减去行均值
def mean_wise(X):
    return (X.T - np.mean(X, axis=1)).T

def pca(X_train, X_test, n):
    pca_trans = PCA(n_components=n).fit(X_train)
    return pca_trans.transform(X_train), pca_trans.transform(X_test)


print("loading data ... ")
X_train, y_train = get_mstar_data("train", 100, 100, 72)
X_test, y_test = get_mstar_data("test", 100, 100, 72)
X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1] * X_train.shape[2]])
X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1] * X_test.shape[2]])
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print("shuffling data ... ")
X_train, y_train = data_shuffle(X_train, y_train)
X_test, y_test = data_shuffle(X_test, y_test)

print("data preprocessing ...")
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = mean_wise(X_train)
X_test = mean_wise(X_test)
# print(X_train)
X_train, X_test = pca(X_train, X_test, 80)

print("training ...")
clf=SVC(C=2.0, kernel='rbf', max_iter=-1, random_state=0)
clf.fit(X_train, y_train)     # 96.82%
print("testing ...")
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


y_pred = clf.predict(X_test)
target_names = ["2S1", "SN_132", "BRDM_2", "BTR_60", "SN_9563", "D7", "T62", "SN_C71", "ZIL131", "ZSU_23_4"]
print(classification_report(y_test, y_pred, target_names=target_names))