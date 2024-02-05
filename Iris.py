# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd
import os, cv2, random
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications.vgg19 import VGG19
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ANSI Escape Sequences (colors)
class bcolors:
    CYAN      = '\033[36m'
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

data = pd.read_csv("archive/full_df.csv")

# Clear Command
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

cls()

def has_condn(term, text): return 1 if term in text else 0

def procDataset(data):
    data["left_cataract"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("cataracte",x))
    data["right_cataract"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("cataracte",x))
    
    data["LD"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy",x))
    data["RD"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy",x))

    data["LG"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("glaucome",x))
    data["RG"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("glaucome",x))
    
    data["LH"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive",x))
    data["RH"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive",x))

    data["LM"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("myopie",x))
    data["RM"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("myopie",x))
    
    data["LA"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration",x))
    data["RA"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration",x))
    
    data["LO"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("drusen",x))
    data["RO"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("drusen",x))
    
    cataracteLeft = data.loc[(data.C ==1) & (data.left_cataract == 1)]["Left-Fundus"].values
    cataracteRight = data.loc[(data.C == 1) & (data.right_cataract == 1)]["Right-Fundus"].values
    
    myopieLeft = data.loc[(data.C == 0) & (data.LM == 1)]["Left-Fundus"].values
    myopieRight = data.loc[(data.C == 0) & (data.RM == 1)]["Right-Fundus"].values 

    glaucomeLeft = data.loc[(data.C == 0) & (data.LG == 1)]["Left-Fundus"].values
    glaucomeRight = data.loc[(data.C == 0) & (data.RG == 1)]["Right-Fundus"].values 

    diabeteLeft = data.loc[(data.C == 0) & (data.LD == 1)]["Left-Fundus"].values
    diabeteRight = data.loc[(data.C == 0) & (data.RD == 1)]["Right-Fundus"].values 
    
    hypertensionLeft = data.loc[(data.C == 0) & (data.LH == 1)]["Left-Fundus"].values
    hypertensionRight = data.loc[(data.C == 0) & (data.RH == 1)]["Right-Fundus"].values 

    normalLeft = data.loc[(data.C == 0) & (data["Left-Diagnostic Keywords"] == "normal fundus")]['Left-Fundus'].sample(350,random_state=42).values
    normalRight = data.loc[(data.C == 0) & (data["Right-Diagnostic Keywords"] == "normal fundus")]['Right-Fundus'].sample(350,random_state=42).values
    
    ageLeft = data.loc[(data.C == 0) & (data.LA == 1)]["Left-Fundus"].values
    ageRight = data.loc[(data.C == 0) & (data.RA == 1)]["Right-Fundus"].values 
    
    autreLeft = data.loc[(data.C == 0) & (data.LO == 1)]["Left-Fundus"].values
    autreRight = data.loc[(data.C == 0) & (data.RO == 1)]["Right-Fundus"].values 
    
    normal = np.concatenate((normalLeft,normalRight),axis = 0);
    cataracte = np.concatenate((cataracteLeft,cataracteRight),axis = 0);
    diabete = np.concatenate((diabeteLeft,diabeteRight),axis = 0);
    glaucome = np.concatenate((glaucomeLeft,glaucomeRight),axis = 0);
    hypertention = np.concatenate((hypertensionLeft,hypertensionRight),axis = 0);
    myopie = np.concatenate((myopieLeft,myopieRight),axis = 0);
    age = np.concatenate((ageLeft,ageRight),axis=0);
    autre = np.concatenate((autreLeft,autreRight),axis = 0);

    return normal, cataracte, diabete, glaucome, hypertention, myopie, age, autre

normal, cataracte, diabete, glaucome, hypertention, myopie, age, autre = procDataset(data);

dataset_dir, labels, dataset = "", [], []

# CREATION OF MODEL DATA
def modelGen(imagecategory, label):
    for img in tqdm(imagecategory,
                    desc = f"{bcolors.OKGREEN}  >{bcolors.ENDC} Generating model data... "):
        imgpath = os.path.join('archive/preprocessed_images', img);
        try: image = cv2.imread(imgpath, cv2.IMREAD_COLOR) ; image = cv2.resize(image, (224, 224))
        except: continue
        dataset.append([np.array(image), np.array(label)])
    random.shuffle(dataset)
    
    return dataset

dataset = modelGen(normal, 0) ; dataset = modelGen(cataracte, 1)
dataset = modelGen(diabete, 2) ; dataset = modelGen(glaucome, 3)
dataset = modelGen(hypertention, 4) ; dataset = modelGen(myopie, 5)
dataset = modelGen(age, 6) ; dataset = modelGen(autre, 7)

train_x = np.array([i[0] for i in dataset]).reshape(-1, 224, 224, 3);
train_y = np.array([i[1] for i in dataset])

# ON SPLIT LE DATASET
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

y_train_cat = to_categorical(y_train,num_classes=8)

y_test_cat = to_categorical(y_test,num_classes=8)

cls()

# CREATION OF THE MODEL

vgg = VGG19(weights="imagenet", include_top = False, input_shape=(224, 224, 3))

for lay in vgg.layers: lay.trainable = False

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="softmax"))

model.add(Dropout(0.3))

model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train_cat, batch_size=32, epochs=10)

cls()
print("""

  ██╗██████╗ ██╗███████╗
  ██║██╔══██╗██║██╔════╝
  ██║██████╔╝██║███████╗
  ██║██╔══██╗██║╚════██║
  ██║██║  ██║██║███████║
  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝
  AI - Ocular Disease Recognition                 
""")
# print(history)

loss, accuracy = model.evaluate(x_test, y_test_cat) 
print(f"{bcolors.OKGREEN}  >{bcolors.ENDC} Accuracy  : {accuracy}")

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model loss and accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['accuracy','loss'], loc='upper right')
plt.show()

y_pred = []
for i in model.predict(x_test): y_pred.append(np.argmax(np.array(i)).astype("int32"))

print(y_pred)
print(accuracy_score(y_test, y_pred))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# MODEL EVALUATION
scores = []
for indexT, indexTest in tqdm(kfold.split(train_x),
    desc = f"{bcolors.OKGREEN}  >{bcolors.ENDC} Model evaluation... "):
    X_train, X_test = train_x[indexT], train_x[indexTest]
    y_train = to_categorical(train_y[indexT], num_classes=8)
    y_test = to_categorical(train_y[indexTest], num_classes=8)

    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    score = model.evaluate(X_test, y_test, verbose=0)
    scores.append(score[1])

# PERFORMANCES
score1 = np.mean(scores)
score2 = np.std(scores)
print(f"{bcolors.OKGREEN}  >{bcolors.ENDC} Accuracy : {score1*100:.2f}% (+/- {score2*100:.2f}%)")
