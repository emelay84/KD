# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
#from sklearn.preprocessing import OneHotEncode



# Create the teacher

def create_teacher(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2):
    recupereTeacherLossAccurayTest= []
    #    for i in range (6):
    teacher = tf.keras.Sequential(
      [
    tf.keras.layers.InputLayer(input_shape),
    tf.keras.layers.Conv1D(filters, kernel_size=8, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Conv1D(filters2, kernel_size=5, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.Conv1D(filters, kernel_size=3, padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(nb_classes),
    #output_layer = layer = tf.keras.layers.Softmax()(gap)
    
    #output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)
    
    #return keras.models.Model(inputs=input_layer, outputs=layer1)
      ],
      name="teacher",
      )
    teacher.summary()
    
      # Train teacher as usual
    
    callbacks = [
    keras.callbacks.ModelCheckpoint(
    f"{output_directory}/best_model_teacher.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
      ]
    teacher.compile(
    optimizer=keras.optimizers.Adam(),
    #      loss='categorical_crossentropy',
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    )
    
    # Train and evaluate teacher on data
    batch_size = 16   
    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))
    
    history= teacher.fit(x_train, y_train,batch_size=mini_batch_size, epochs=1,validation_data=(x_test,y_test),
    callbacks=callbacks,)
    
    #history= teacher.fit(x_train, y_train,batch_size=batch_size, epochs=200,
    #callbacks=callbacks, validation_data=(x_val, y_val),)
    resultat = teacher.evaluate(x_test, y_test)
    recupereTeacherLossAccurayTest.append(resultat)
     #teacher = keras.models.load_model(f"best_model_teacher.h5")
    histo_dfteacher = pd.DataFrame(history.history)
    hist_csv_file = output_directory + '/historyteacher' + '.csv'
    with open(hist_csv_file,mode='w' ) as f:
        histo_dfteacher.to_csv(f)
    #test_loss, test_acc = teacher.evaluate(x_test, y_test)
    loss =history.history['loss'] 
    #val_loss =history.history['val_loss']
    plt.figure()
    plt.plot(loss,'b',label ='Validation')
    plt.plot(loss,'bo',label ='Entrainement')
    plt.title(f"{output_directory}/best_model_teacher.h5")
    plt.xlabel("epoch")
    plt.ylabel('')
    plt.legend()
    plt.savefig(output_directory + 'figteacherfold'+'.png')
    plt.show()
    plt.close()
    
    
    metric = "categorical_accuracy"
    plt.figure()
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.savefig(output_directory + 'figteacherfold1'+'.png')
    plt.show()
    plt.close()
    
    tech=recupereTeacherLossAccurayTest
    np.savetxt(output_directory + 'tech.out',tech,delimiter='\t')      