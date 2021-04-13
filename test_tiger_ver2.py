#! pip3 install focal-loss
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Masking, Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, concatenate, UpSampling2D
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import LSTM
#from tensorflow.data.Dataset import from_tensor_slices
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
# from focal_loss import BinaryFocalLoss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from sklearn.metrics import matthews_corrcoef
#from keras.utils.generic_utils import get_custom_objects
import sys
import json
import datetime
import sklearn
import pickle

class F1Callback(Callback):
    def __init__(self, model, val, gt, pred_one=False):
        self.model = model
        self.val = val
        self.gt = gt
        self.n = pred_one
        
    def on_epoch_end(self, epoch, logs):
        
        if self.n == False:
            pred_pur, *pred = model.predict(self.val)
        else: 
            pred_pur = model.predict(self.val)
        p = np.round(np.array(pred_pur).reshape(-1))
        print(p, pred_pur)
        
        f1_val = f1_score(self.gt, p)
        print("f1_val =", f1_val)
        print(confusion_matrix(self.gt, p))

class MCC_Callback(Callback):
    def __init__(self, model, val, gt, pred_one=False):
        self.model = model
        self.val = val
        self.gt = gt
        self.n = pred_one
        
    def on_epoch_end(self, epoch, logs):
        
        if self.n == False:
            pred_pur, *pred = model.predict(self.val)
        else: 
            pred_pur = model.predict(self.val)
        p = np.round(np.array(pred_pur).reshape(-1))
        mcc = matthews_corrcoef(self.gt, p)
        print("MCC val =", mcc)
        print(confusion_matrix(self.gt, p))


def preprocess_image(image):
    img = tf.image.decode_jpeg(image, channels=3)
    #img = tf.image.resize(img, [224, 224])
    img = tf.image.resize(img, [112, 112])
    img = tf.cast(img, tf.float32) / 255.0  # normalize to [0,1] range
    return img
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    #img = tf.read_file(path)
    return preprocess_image(img)
def like_relu_function(x):
    return tf.math.minimum(tf.math.maximum(1.0, x), 4.0)
#get_custom_objects().update({'custom_activation': Activation(like_relu_function)})


def get_model_origin(vector_shape=(47,)):
    input1 = Input(shape=vector_shape)
    
    # NN architecture using input1: user and item data
    x1 = Dense(256, input_dim=vector_shape, activation='relu')(input1)
    
    # ouptut 1 purchase
    y1 = Dense(256, activation="relu")(x1)
    y1 = Dense(1, activation="sigmoid", name='purchased')(x1)
    
    return Model(inputs=input1, outputs=y1)



def lambda_handler(event, context):
    #event = json.loads(event['body'])
    rates =  np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    
    item_num = len(event['item_data'].keys())
    find_num = len(rates)
    item_sku_ids = event['item_data'].keys()
    length_coupon_rate = len(event['coupon_discount_rate'])
    length_coupon_price = len(event['coupon_discount_price'])
    
    
    #user_features
    d = datetime.datetime.now()
    registered_monthly_date = datetime.datetime.strptime(event['registered_monthly_date'], '%Y-%m-%d %H:%M:%S')
    date_of_birth = datetime.datetime.strptime(event['date_of_birth'], '%Y-%m-%d %H:%M:%S')
    keizoku = (d - registered_monthly_date) / datetime.timedelta(days=1)
    ave_purchase_count = event['purchase_cnt'] / keizoku
    age = (d - date_of_birth).days / 366
    coupon_count = length_coupon_price + length_coupon_rate
    
    plan_id = [0,0,0]
    if event['plan_id'] == 1:
        plan_id = [1,0,0]
    elif event['plan_id'] == 2:
        plan_id = [0,1,0]
    elif event['plan_id'] == 4:
        plan_id = [0,0,1]
        
    payment_type = [0,0]
    if event['payment_type'] == 1:
        payment_type = [1,0]
    elif event['payment_type'] == 2:
        payment_type = [0,1]
        
    user_features = np.hstack(([keizoku], [event['user_fi_count']], [ave_purchase_count], [age], [coupon_count], plan_id, payment_type))
    
    #item_featuers
    df = pd.DataFrame(event['item_data']).T
    sale = df[['max_sale_price', 'max_sale_rate']].values
    is_sale = [[len(np.hstack(i)) > 0] for i in sale]
    
    season = pd.DataFrame(list(df['seasons'].values))
    season = season[['winter', 'summer', 'late_autumn', 'early_autumn', 'late_spring', 'early_spring']].values
    
    category = np.zeros((item_num, 8))
    category_id = df['category_id'].values
    for n,i in enumerate(category_id):
        if (i < 8 and i > 0) or i == 101:
            if i != 101:
                category[n][i-1] = 1
            else:
                category[n][7] = 1
    #category = category[:, [0,3,4,2,5,6]]
    
    ac_color = np.zeros((item_num, 18))
    ac_color_ = df['ac_color'].values
    for n,i in enumerate(ac_color_):
        if i < 19 and i > 0:
            ac_color[n][i-1] = 1
    
    item_features = np.hstack((df[['retail_price', 'item_fi_count']].values, is_sale, season, category, ac_color))
    
    #price_features   - detail in "https://drive.google.com/file/d/1aMDtg42hD8ZQlZBlAhbaLWhsHAJL5K0j/view?usp=sharing"
    retail_price = df['retail_price'].values
    
    P = np.dot(retail_price.reshape(-1,1), rates[np.newaxis]) #member_special_price
    Sp = P - np.array([[i for j in range(find_num)] for i in df['max_sale_price'].values])
    Sr = P * (1 - np.array([[i for j in range(find_num)] for i in df['max_sale_rate'].values]))
    S = np.maximum(Sp, Sr)
    
    if length_coupon_rate > 0:
        ncr = len(event['coupon_discount_rate'])
        Scr = np.transpose([S for i in range(ncr)], (1,2,0))
        Cr = 1 - np.array([[event['coupon_discount_rate'] for j in range(find_num)] for i in range(item_num)])
        Pr = Scr * Cr
        
    if length_coupon_price > 0:
        ncp = len(event['coupon_discount_price'])
        Scp = np.transpose([S for i in range(ncp)], (1,2,0))
        Cp = np.array([[event['coupon_discount_price'] for j in range(find_num)] for i in range(item_num)])
        Pp = Scp - Cp
    
    if length_coupon_rate > 0 and length_coupon_price > 0:
        PP = np.concatenate((Pr, Pp), axis=2)
    elif length_coupon_rate > 0:
        PP = Pr
    elif length_coupon_price > 0:
        PP = Pp
    else:
        PP = S[:,:,np.newaxis]
    
    P_max = np.max(PP, axis=2)[:,:,np.newaxis]
    P_min = np.min(PP, axis=2)[:,:,np.newaxis]
    # P_var = np.var(P, axis=2)[:,:,newaxis]
    P_avg = np.average(PP, axis=2)[:,:,np.newaxis]
    
    price_features = np.concatenate((P_max, P_min, P_avg, S[:,:,np.newaxis], P[:,:,np.newaxis]), axis=2)
    # item_num * find_num
    
    
    # concatenate
    user_features = [[user_features for j in range(find_num)] for i in range(item_num)]
    item_features = np.transpose([item_features for i in range(find_num)], (1,0,2))
    features = np.concatenate((user_features, item_features, price_features), axis=2).astype(np.float32)

    
    key = 'scaler_tiger_ver2.pkl'
    with open(key, 'rb') as pickle_file:
        scaler = pickle.load(pickle_file)
    arr = features.reshape(-1, features.shape[-1])
    payload = scaler.transform(arr).astype(float)
    
    model = get_model_origin(vector_shape=(payload.shape[1],))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

    model.load_weights(f'origin_checkpoint/0001.ckpt')
    print(payload.shape)
    result = model.predict(payload)
    
    result = np.array(result).reshape(-1, find_num).astype(float) * P.reshape(-1,find_num)
    m = np.argmax(result, axis=1)
    result_price = [features[i,j,-1] for i,j in enumerate(m)]
    
    price_hash = {}
    for i,j in zip(item_sku_ids, result_price):
        price_hash[str(i)] = str(int(j))
    
    print(price_hash)
    return json.dumps({
                "price_hash": price_hash,
                "item_rate_set": {
                    "type": 3,
                    "version": 10
                }
            })

def tmp_price_prediction():
    lambda_handler(json.loads(sys.argv[1]), '')

if __name__ == "__main__":
    print(lambda_handler(json.loads(sys.argv[1]), ''))