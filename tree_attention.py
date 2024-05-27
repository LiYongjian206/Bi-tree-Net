import numpy as np
from keras import Input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Add, Multiply, \
    GlobalAveragePooling1D, Concatenate, Reshape, BatchNormalization, ELU, DepthwiseConv1D, AveragePooling1D, \
    Conv2D, GlobalMaxPool1D, Conv1DTranspose, Bidirectional, GRU, LSTM, Dropout
from keras import backend as K
from keras.models import Model
from keras_flops import get_flops

# 学习率更新以及调整

def scheduler(epoch):

    if epoch == 0:
        lr = K.get_value(model.optimizer.lr)  # keras默认0.001
        K.set_value(model.optimizer.lr, lr*10)
        print("lr changed to {}".format(lr))
    if epoch != 0:
    # else:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr / (1 + 0.001 * epoch))
        # K.set_value(model.optimizer.lr, lr * math.pow(0.99, epoch))
        # K.set_value(model.optimizer.lr, lr / (10))
        # print("lr changed to {}".format(model.optimizer.lr))

    return K.get_value(model.optimizer.lr)

index = 7
file = 'N'

# 数据导入
data1 = np.load('D:/BP-INTER/data/' + file + '/train_ppg' + str(index) + '.npy', allow_pickle=True)
data2 = np.load('D:/BP-INTER/data/' + file + '/train_ecg' + str(index) + '.npy', allow_pickle=True)
data3 = np.load('D:/BP-INTER/data/' + file + '/train_abp' + str(index) + '.npy', allow_pickle=True)

# data3 = data3[:, 1]
label1 = data3[:, 0]
label2 = data3[:, 1]
label3 = data3[:, 2]
i = 125
o = 1

def DC_Block1(input, k, c):

    conv1 = DepthwiseConv1D(kernel_size=k, strides=1)(input)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = ELU()(conv1)

    conv1 = Conv1D(filters=c, kernel_size=1, strides=1)(conv1)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = ELU()(conv1)

    return conv1

def Filter(inputs, c):

    x1 = GlobalAveragePooling1D()(inputs)
    x2 = GlobalMaxPool1D()(inputs)
    x = Concatenate()([x1, x2])
    x = Reshape((2, int(inputs.shape[-1]), 1), input_shape=(None, 2 * int(inputs.shape[-1])))(x)
    x = Conv2D(filters=1, kernel_size=(2, 1), strides=1)(x)
    x = BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = ELU()(x)
    x = Flatten()(x)

    x = Dense(int(x.shape[-1]), activation='relu')(x)  # 调节参数r 4、8、16、32
    x = Dense(int(inputs.shape[-1]), activation='hard_sigmoid')(x)
    x = Multiply()([inputs, x])
    x = Flatten()(x)
    x = Dense(c, activation='relu')(x)
    x = Dense(c, activation='softmax')(x)

    y = Flatten()(inputs)
    y1 = Reshape((int(inputs.shape[1]), int(inputs.shape[-1]), 1),
                 input_shape=(None, int(inputs.shape[1]) * int(inputs.shape[-1])))(y)
    y2 = Reshape((int(inputs.shape[1]), int(inputs.shape[-1]), 1),
                 input_shape=(None, int(inputs.shape[1]) * int(inputs.shape[-1])))(y)

    z1 = Multiply()([y1, y2])
    z1 = Conv1D(filters=8, kernel_size=1, strides=1)(z1)  # 升维
    z1 = BatchNormalization(momentum=0.99, epsilon=0.001)(z1)
    z1 = ELU()(z1)
    z2 = Add()([y1, y2])
    z2 = Conv1D(filters=8, kernel_size=1, strides=1)(z2)  # 升维
    z2 = BatchNormalization(momentum=0.99, epsilon=0.001)(z2)
    z2 = ELU()(z2)

    z = Add()([z1, z2])
    z = Flatten()(z)
    z = Dense(c, activation='softmax')(z)

    return x + z

def RF(inputs1):

    input1 = Conv1D(filters=128, kernel_size=7, strides=1)(inputs1)
    input1 = BatchNormalization(momentum=0.99, epsilon=0.001)(input1)
    input1 = ELU()(input1)
    # input1 = MaxPooling1D(pool_size=2, strides=2)(input1)

    # input2 = Conv1D(filters=128, kernel_size=3, strides=1)(inputs2)
    # input2 = BatchNormalization(momentum=0.99, epsilon=0.001)(input2)
    # input2 = ELU()(input2)
    # input2 = MaxPooling1D(pool_size=2, strides=2)(input2)

    # inputs = Add()([input1, input2])

    # 第1层
    RF1 = DC_Block1(input1, 3, 128)
    RF1 = MaxPooling1D(pool_size=2, strides=2)(RF1)

    # 第2层
    RF21 = DC_Block1(RF1, 3, 64)
    RF22 = DC_Block1(RF1, 5, 64)
    RF21 = MaxPooling1D(pool_size=2, strides=2)(RF21)
    RF22 = MaxPooling1D(pool_size=2, strides=2)(RF22)

    # 第3层
    RF31 = DC_Block1(RF21, 3, 32)
    RF32 = DC_Block1(RF21, 5, 32)
    RF33 = DC_Block1(RF22, 3, 32)
    RF34 = DC_Block1(RF22, 5, 32)
    RF31 = MaxPooling1D(pool_size=2, strides=2)(RF31)
    RF32 = MaxPooling1D(pool_size=2, strides=2)(RF32)
    RF33 = MaxPooling1D(pool_size=2, strides=2)(RF33)
    RF34 = MaxPooling1D(pool_size=2, strides=2)(RF34)

    # 第4层
    RF41 = DC_Block1(RF31, 3, 16)
    RF41 = MaxPooling1D(pool_size=2, strides=2)(RF41)
    RF42 = DC_Block1(RF31, 5, 16)
    RF42 = MaxPooling1D(pool_size=2, strides=2)(RF42)
    RF43 = DC_Block1(RF32, 3, 16)
    RF43 = MaxPooling1D(pool_size=2, strides=2)(RF43)
    RF44 = DC_Block1(RF32, 5, 16)
    RF44 = MaxPooling1D(pool_size=2, strides=2)(RF44)
    RF45 = DC_Block1(RF33, 3, 16)
    RF45 = MaxPooling1D(pool_size=2, strides=2)(RF45)
    RF46 = DC_Block1(RF33, 5, 16)
    RF46 = MaxPooling1D(pool_size=2, strides=2)(RF46)
    RF47 = DC_Block1(RF34, 3, 16)
    RF47 = MaxPooling1D(pool_size=2, strides=2)(RF47)
    RF48 = DC_Block1(RF34, 5, 16)
    RF48 = MaxPooling1D(pool_size=2, strides=2)(RF48)

    # Attention
    a = 16
    x1 = Filter(RF41, a)
    x2 = Filter(RF42, a)
    x3 = Filter(RF43, a)
    x4 = Filter(RF44, a)
    x5 = Filter(RF45, a)
    x6 = Filter(RF46, a)
    x7 = Filter(RF47, a)
    x8 = Filter(RF48, a)

    # g1 = GRU(a, activation='tanh')(RF41)
    # g2 = GRU(a, activation='tanh')(RF42)
    # g3 = GRU(a, activation='tanh')(RF43)
    # g4 = GRU(a, activation='tanh')(RF44)
    # g5 = GRU(a, activation='tanh')(RF45)
    # g6 = GRU(a, activation='tanh')(RF46)
    # g7 = GRU(a, activation='tanh')(RF47)
    # g8 = GRU(a, activation='tanh')(RF48)
    #
    # x1 = x1 + g1
    # x2 = x2 + g2
    # x3 = x3 + g3
    # x4 = x4 + g4
    # x5 = x5 + g5
    # x6 = x6 + g6
    # x7 = x7 + g7
    # x8 = x8 + g8
    #
    y1 = Multiply()([RF41, x1])
    y2 = Multiply()([RF42, x2])
    y3 = Multiply()([RF43, x3])
    y4 = Multiply()([RF44, x4])
    y5 = Multiply()([RF45, x5])
    y6 = Multiply()([RF46, x6])
    y7 = Multiply()([RF47, x7])
    y8 = Multiply()([RF48, x8])

    TC = 16

    z1 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y1)
    # z1 = Dropout(0.5)(z1)
    z1 = BatchNormalization(momentum=0.99, epsilon=0.001)(z1)
    z1 = ELU()(z1)
    # z1 = Conv1DTranspose(filters=TC, kernel_size=5, strides=1)(z1)
    # z1 = Dropout(0.5)(z1)
    # z1 = BatchNormalization(momentum=0.99, epsilon=0.001)(z1)
    # z1 = ELU()(z1)
    z1 = GlobalAveragePooling1D()(z1)

    z2 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y2)
    # z2 = Dropout(0.5)(z2)
    z2 = BatchNormalization(momentum=0.99, epsilon=0.001)(z2)
    z2 = ELU()(z2)
    # z2 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(z2)
    # z2 = Dropout(0.5)(z2)
    # z2 = BatchNormalization(momentum=0.99, epsilon=0.001)(z2)
    # z2 = ELU()(z2)
    z2 = GlobalAveragePooling1D()(z2)

    z3 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y3)
    # z3 = Dropout(0.5)(z3)
    z3 = BatchNormalization(momentum=0.99, epsilon=0.001)(z3)
    z3 = ELU()(z3)
    # z3 = Conv1DTranspose(filters=TC, kernel_size=5, strides=1)(z3)
    # z3 = Dropout(0.5)(z3)
    # z3 = BatchNormalization(momentum=0.99, epsilon=0.001)(z3)
    # z3 = ELU()(z3)
    z3 = GlobalAveragePooling1D()(z3)

    z4 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y4)
    # z4 = Dropout(0.5)(z4)
    z4 = BatchNormalization(momentum=0.99, epsilon=0.001)(z4)
    z4 = ELU()(z4)
    # z4 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(z4)
    # z4 = Dropout(0.5)(z4)
    # z4 = BatchNormalization(momentum=0.99, epsilon=0.001)(z4)
    # z4 = ELU()(z4)
    z4 = GlobalAveragePooling1D()(z4)

    z5 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y5)
    # z5 = Dropout(0.5)(z5)
    z5 = BatchNormalization(momentum=0.99, epsilon=0.001)(z5)
    z5 = ELU()(z5)
    # z5 = Conv1DTranspose(filters=TC, kernel_size=5, strides=1)(z5)
    # z5 = Dropout(0.5)(z5)
    # z5 = BatchNormalization(momentum=0.99, epsilon=0.001)(z5)
    # z5 = ELU()(z5)
    z5 = GlobalAveragePooling1D()(z5)

    z6 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y6)
    # z6 = Dropout(0.5)(z6)
    z6 = BatchNormalization(momentum=0.99, epsilon=0.001)(z6)
    z6 = ELU()(z6)
    # z6 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(z6)
    # z6 = Dropout(0.5)(z6)
    # z6 = BatchNormalization(momentum=0.99, epsilon=0.001)(z6)
    # z6 = ELU()(z6)
    z6 = GlobalAveragePooling1D()(z6)

    z7 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y7)
    # z7 = Dropout(0.5)(z7)
    z7 = BatchNormalization(momentum=0.99, epsilon=0.001)(z7)
    z7 = ELU()(z7)
    # z7 = Conv1DTranspose(filters=TC, kernel_size=5, strides=1)(z7)
    # z7 = Dropout(0.5)(z7)
    # z7 = BatchNormalization(momentum=0.99, epsilon=0.001)(z7)
    # z7 = ELU()(z7)
    z7 = GlobalAveragePooling1D()(z7)

    z8 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(y8)
    # z8 = Dropout(0.5)(z8)
    z8 = BatchNormalization(momentum=0.99, epsilon=0.001)(z8)
    z8 = ELU()(z8)
    # z8 = Conv1DTranspose(filters=TC, kernel_size=3, strides=1)(z8)
    # z8 = Dropout(0.5)(z8)
    # z8 = BatchNormalization(momentum=0.99, epsilon=0.001)(z8)
    # z8 = ELU()(z8)
    z8 = GlobalAveragePooling1D()(z8)

    z = Concatenate()([z1, z2, z3, z4, z5, z6, z7, z8])

    return z

def models(inputs1, inputs2):

    x = RF(inputs1)
    y = RF(inputs2)

    z = Concatenate()([x, y])

    z = Dense(128, activation='linear')(z)
    z = Dropout(0.5)(z)
    z = Dense(128, activation='linear')(z)
    z = Dropout(0.5)(z)
    # z = Dense(o, activation='linear')(z)
    z1 = Dense(o, activation='linear')(z)
    z2 = Dense(o, activation='linear')(z)
    z3 = Dense(o, activation='linear')(z)

    # y1 = Dense(128, activation='linear')(z)
    # y1 = Dropout(0.5)(y1)
    # y1 = Dense(128, activation='linear')(y1)
    # y1 = Dropout(0.5)(y1)
    # z1 = Dense(o, activation='linear')(y1)
    #
    # y2 = Dense(128, activation='linear')(z)
    # y2 = Dropout(0.5)(y2)
    # y2 = Dense(128, activation='linear')(y2)
    # y2 = Dropout(0.5)(y2)
    # z2 = Dense(o, activation='linear')(y2)
    #
    # y3 = Dense(128, activation='linear')(z)
    # y3 = Dropout(0.5)(y3)
    # y3 = Dense(128, activation='linear')(y3)
    # y3 = Dropout(0.5)(y3)
    # z3 = Dense(o, activation='linear')(y3)

    out = Model(inputs=[inputs1, inputs2], outputs=[z1, z2, z3], name="model")

    return out

inputs1 = Input(shape=(i, 1))
inputs2 = Input(shape=(i, 1))
model = models(inputs1, inputs2)
model.summary()
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 6:.05} M")

# root_mean_square_error
def root_mean_square_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true)))

model.compile(loss=['mean_absolute_error', 'mean_absolute_error', 'mean_absolute_error'],
              loss_weights=[0.25, 0.5, 0.25], optimizer='Adam')
# loss_weights=[0.2, 0.6, 0.2],
# loss_weights=[0.25, 0.5, 0.25],
# model.compile(loss='mean_absolute_error', optimizer='Adam')

# mean_squared_error 均方误差
# mean_absolute_error 平均绝对误差
# mean_absolute_percentage_error 平均绝对百分比误差
# mean_squared_logarithmic_error（MSLE）均方根对数误差

# filepath = 'D:/BP-INTER/data/' + file + '/model' + str(index) + '.hdf5'  # 保存模型的路径
# filepath = 'D:/BP-INTER/data/model' + str(index) + '.hdf5'
# checkpoint = ModelCheckpoint(filepath=filepath, verbose=2,
#                              monitor='val_loss', mode='min', save_best_only='True')
checkpoint = ModelCheckpoint(filepath='C:/Users/lyj/Desktop/test.hdf5', verbose=2,
                             monitor='val_loss', mode='min', save_best_only='True')
reduce_lr = LearningRateScheduler(scheduler)  # 学习率的改变
callback_lists = [checkpoint, reduce_lr]

# v1 = np.load('D:/BP-INTER/data/' + file + '/test_ppg' + str(index) + '.npy', allow_pickle=True)
# v2 = np.load('D:/BP-INTER/data/' + file + '/test_ecg' + str(index) + '.npy', allow_pickle=True)
# v3 = np.load('D:/BP-INTER/data/' + file + '/test_abp' + str(index) + '.npy', allow_pickle=True)
# validation_data=[[v1, v2], v3]
# validation_data=[[v1, v2], [v3[:, 0], v3[:, 1], v3[:, 2]]],
# validation_split=0.1

train_history = model.fit(x=[data1, data2],
                          y=[label1, label2, label3], verbose=2,
                          validation_split=0.1,
                          class_weight=None, callbacks=callback_lists,
                          epochs=50, batch_size=512, shuffle=True)
# label1, label2, label3
# v1 = np.load('D:/BP-INTER/data/' + file + '/test_ppg' + str(index) + '.npy', allow_pickle=True)
# v2 = np.load('D:/BP-INTER/data/' + file + '/test_ecg' + str(index) + '.npy', allow_pickle=True)
# v3 = np.load('D:/BP-INTER/data/' + file + '/predict' + str(index) + '.npy', allow_pickle=True)
# validation_data=[[v1, v2], v3]