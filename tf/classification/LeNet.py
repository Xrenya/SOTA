import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Activation
from tensorflow.keras import Model

class Linear(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Linear, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='he_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class LeNetCustom(Model):
    def __init__(self, num_classes: int=10):
        super(LeNet, self).__init__()
        self.conv_1 = Conv2D(filters=6,
                             kernel_size=5,
                             strides=1,
                             padding='valid',
                             activation='relu')
        self.pool_1 = AveragePooling2D(pool_size=2,
                                       strides=2,
                                       padding='valid')
        self.conv_2= Conv2D(filters=16,
                             kernel_size=5,
                             strides=1,
                             padding='valid',
                             activation='relu')
        self.pool_2 = AveragePooling2D(pool_size=2,
                                       strides=2,
                                       padding='valid')
        self.fc_1 = Linear(120)
        self.fc_2 = Linear(84)
        self.fc_3 = Linear(num_classes)
        
    def call(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = Activation('relu')(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = self.fc_1(x)
        x = Activation('relu')(x)
        x = self.fc_2(x)
        x = Activation('relu')(x)
        return self.fc_3(x)

class LeNet(Model):
    def __init__(self, num_classes: int=10):
        super(LeNet, self).__init__()
        self.conv_1 = Conv2D(filters=6,
                             kernel_size=5,
                             strides=1,
                             padding='valid',
                             activation='relu')
        self.pool_1 = AveragePooling2D(pool_size=2,
                                       strides=2,
                                       padding='valid')
        self.conv_2= Conv2D(filters=16,
                             kernel_size=5,
                             strides=1,
                             padding='valid',
                             activation='relu')
        self.pool_2 = AveragePooling2D(pool_size=2,
                                       strides=2,
                                       padding='valid')
        self.fc_1 = Dense(units=120, activation='relu')
        self.fc_2 = Dense(units=84, activation='relu')
        self.fc_3 = Dense(units=num_classes)
        
    def call(self, x):
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = Flatten()(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return self.fc_3(x)
