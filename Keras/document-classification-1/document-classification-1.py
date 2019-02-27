# -*- coding: utf-8 -*-
import numpy as np
import sys
import os
import heapq
import glob
import collections
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer

###
# 用意したデータを読み込むフェーズです。
###
# 読み込むファイルの名前
file = "data.txt"

# 各行のラベルを保持するリスト
labels = []

# 各行のデータ部を保持するリスト
words_arr  = []

with open(file) as f:
  for line in f:

    # 行をカンマで分割し、ラベル部とデータ部を分割します
    blocks = line.rstrip().split(',',1)

    # ラベル部, データ部 それぞれをリストに追加します
    labels.append(blocks[0])
    words_arr.append(blocks[1])

###
# データ部・ラベル部をそれぞれベクトル化します。
###

# データ部をベクトル化します
tokenizer = Tokenizer()
tokenizer.fit_on_texts(words_arr)
x_data = tokenizer.texts_to_matrix(words_arr, "binary")

# ラベル部をベクトル化します
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
y_data = label_tokenizer.texts_to_matrix(labels, "binary")

###
# 学習用データと検証用データにデータを分割します。
# ここでは 9:1 の割合に分割します。
###
train_size = int(len(x_data) * 0.9)
x_train,x_test = x_data[:train_size],x_data[train_size:]
y_train,y_test = y_data[:train_size],y_data[train_size:]

##
# これから作り出す脳みその形を宣言します。
# InputLayer : モデルへの入力部分
# Dense      : 第1引数で出力の次元数を指定します。入力の次元数はinput_shapeで指定します(指定しない場合は出力と同じ)
##
model = Sequential()
model.add(InputLayer(input_shape=(x_train.shape[1],)))
model.add(Dense(y_train.shape[1], activation='softmax'))

##
# モデルをコンパイルします。
# まだ何も学習していない空っぽの脳みそを生成するイメージです。
##
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

###
# model.fit() を実行することにより、モデルの学習を行います。
###
epochs = 1000
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

###
# 作成したモデル(脳みそ)の性能を評価します。
###
print()
score = model.evaluate(x_test, y_test, verbose=1)
print('■作成したモデルの性能評価')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print()

###
# 最後に「株主総会」という単語を含む文章を想定して、人工知能にカテゴリを推測させます。
# texts を tokenizer で matrix というベクトル化表現に変形させ、
# model.predict() で「推測」を実行します。
# result には、各ラベルごとの「確率」が数列で格納されます。 
###
print('■作成したモデルを実際に利用してみる')
texts = ["株主総会"]
matrix = tokenizer.texts_to_matrix(texts, "binary")
result = model.predict(matrix)
print("入力:", texts)
print(label_tokenizer.word_index)
print(result)
