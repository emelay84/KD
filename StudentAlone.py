#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 16:17:13 2021

@author: ay
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.layers import Conv2D,GlobalAveragePooling2D,Dense,Softmax,Flatten,MaxPooling2D,Dropout,Activation, Lambda, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import kullback_leibler_divergence as KLD_Loss, categorical_crossentropy as logloss
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_accuracy

#from google.colab import files
#from sklearn.preprocessing import OneHotEncoder


# Create the student


def create_StudentAlone(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2):
    recupereStudentAloneLossAccurayTest1=[]

    studentAlone = tf.keras.Sequential(
      [
        tf.keras.layers.InputLayer(input_shape),
        tf.keras.layers.Conv1D(filters=filters, kernel_size=8, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    
        tf.keras.layers.Conv1D(filters=filters2, kernel_size=5, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    
        tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
    
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(nb_classes),
        #output_layer = layer = tf.keras.layers.Softmax()(gap)
    
        #output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
    
        #return keras.models.Model(inputs=input_layer, outputs=layer1)
      ],
      name="studentAlone",
      )
    studentAlone.summary()
      # Clone student for later comparison
      #student_scratch = keras.models.clone_model(student)
    
    
    callbacks = [
    keras.callbacks.ModelCheckpoint(
        f"{output_directory}/best_model_student.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
      ]
    # Train student as doen usually
    studentAlone.compile(
    optimizer=keras.optimizers.Adam(),
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    )
    
    
    # optimizer=keras.optimizers.Adam(),
    #loss='categorical_crossentropy',
    #metrics=['accuracy'],
    
    
    # Train and evaluate student trained from scratch.
    batch_size = 16    
    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

    
    history1= studentAlone.fit(x_train, y_train,batch_size=mini_batch_size, epochs=1,validation_data=(x_test,y_test),
    callbacks=callbacks,)
    
    resultat = studentAlone.evaluate(x_test, y_test)
    recupereStudentAloneLossAccurayTest1.append(resultat)
    histo_dfstudentAlone = pd.DataFrame(history1.history)
    hist_csv_file = output_directory + '/historystudentAlone'+'.csv'
    with open(hist_csv_file,mode='w' ) as f:
        histo_dfstudentAlone.to_csv(f)
    
    
    
    loss1 =history1.history['loss'] 
    val_loss =history1.history['val_loss']
    plt.figure()
    plt.plot(val_loss,'b',label ='Validation')
    plt.plot(loss1,'bo',label ='Entrainement')
    plt.title(f"{output_directory}/best_model_student.h5")
    plt.xlabel("epoch")
    plt.ylabel('')
    plt.legend()
    plt.savefig(output_directory+'/figstAlone1fold'+'.png')
    plt.show()
    plt.close()
    
    
    metric1 = "categorical_accuracy"
    plt.figure()
    plt.plot(history1.history[metric1])
    plt.plot(history1.history["val_" + metric1])
    plt.title("model " + metric1)
    plt.ylabel(metric1, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(output_directory +'/figstAlone1fold1'+'.png')
    plt.show()
    plt.close()
    
    tech1=recupereStudentAloneLossAccurayTest1
    np.savetxt(output_directory +'tech.out',tech1,delimiter='\t')
