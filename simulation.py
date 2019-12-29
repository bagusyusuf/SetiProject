import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import tensorflow.keras
import numpy as np
import cv2
import operator

root = Tk()
root.title("predicion apps")
root.geometry("460x280+300+150")
root.resizable(width=False, height=False)

fileName = {"fileName": ""}
className = {"brightpixel": 0,
             "narrowband": 0,
             "narrowbanddrd": 0,
             "noise": 0,
             "squarepulsednarrowband": 0,
             "squiggle": 0,
             "squigglesquarepulsednarrowband": 0}


def openfn():
    filename = filedialog.askopenfilename(title='open')
    return filename


def open_img():
    x = openfn()
    img2 = Image.open(x)
    img2 = img2.resize((250, 250), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(img2)

    # panel = Label(root, image=img)
    # panel.image = img
    # panel.pack(padx=5, pady=5, side=tk.RIGHT)

    panel.configure(image=img2)
    panel.image = img2
    fileName["fileName"] = x


def predict():

    image = cv2.imread(fileName["fileName"])
    xVal = cv2.resize(image, (224, 224))
    xVal = xVal.astype('float32') / 255
    xVal -= meanTrain
    xVal = np.expand_dims(xVal, axis=0)
    yVal = new_model.predict(xVal)
    # yVal = np.zeros((1, 7))

    float_formatter = "{:.4f}".format

    className["brightpixel"] = yVal[0][0]
    className["narrowband"] = yVal[0][1]
    className["narrowbanddrd"] = yVal[0][2]
    className["noise"] = yVal[0][3]
    className["squarepulsednarrowband"] = yVal[0][4]
    className["squiggle"] = yVal[0][5]
    className["squigglesquarepulsednarrowband"] = yVal[0][6]

    predictLabel_1.delete('1.0', END)
    predictLabel_1.insert(END, float_formatter(yVal[0][0]))

    predictLabel_2.delete('1.0', END)
    predictLabel_2.insert(END, float_formatter(yVal[0][1]))

    predictLabel_3.delete('1.0', END)
    predictLabel_3.insert(END, float_formatter(yVal[0][2]))

    predictLabel_4.delete('1.0', END)
    predictLabel_4.insert(END, float_formatter(yVal[0][3]))

    predictLabel_5.delete('1.0', END)
    predictLabel_5.insert(END, float_formatter(yVal[0][4]))

    predictLabel_6.delete('1.0', END)
    predictLabel_6.insert(END, float_formatter(yVal[0][5]))

    predictLabel_7.delete('1.0', END)
    predictLabel_7.insert(END, float_formatter(yVal[0][6]))

    predictLabel_8.delete('1.0', END)
    predictLabel_8.insert(
        END, max(className.items(), key=operator.itemgetter(1))[0])


lbl = tk.Label(root, text="Simulation", bg="grey", fg="white").pack(fill=tk.X)
btn = Button(root, text='open image',
             command=open_img).place(x=10, y=30 + 10)
#pack(    padx=5, pady=5, side=tk.LEFT)
btn2 = Button(root, text='Predict', command=predict).place(x=100, y=30 + 10)

predictLabel_1 = tk.Text(root, width=6, height=1, wrap=WORD)
predictLabel_1.place(x=10, y=30 + 30 + 20)

predictLabel_2 = tk.Text(root, width=6, height=1, wrap=WORD)
predictLabel_2.place(x=10 + 80, y=30 + 30 + 20)

predictLabel_3 = tk.Text(root, width=6, height=1, wrap=WORD)
predictLabel_3.place(x=10, y=30 + 30 + 50)

predictLabel_4 = tk.Text(root, width=6, height=1, wrap=WORD)
predictLabel_4.place(x=10 + 80, y=30 + 30 + 50)

predictLabel_5 = tk.Text(root, width=6, height=1, wrap=WORD)
predictLabel_5.place(x=10, y=30 + 30 + 80)

predictLabel_6 = tk.Text(root, width=6, height=1, wrap=WORD)
predictLabel_6.place(x=10 + 80, y=30 + 30 + 80)

predictLabel_7 = tk.Text(root, width=6, height=1, wrap=WORD)
predictLabel_7.place(x=10, y=30 + 30 + 110)

meanTrain = np.load('geekfile.npy')
new_model = tensorflow.keras.models.load_model("./model/my_model_89.h5")

predictLabel = tk.Label(root, text="Prediction :").place(x=10, y=30 + 30 + 140)
predictLabel_8 = tk.Text(root, width=15, height=1)
predictLabel_8.place(x=10, y=30 + 30 + 170)


img = Image.new("RGB", (250, 250), "white")
# img = img.resize((250, 250), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)

panel = Label(root, image=img)
panel.image = img
panel.pack(padx=5, pady=5, side=tk.RIGHT)

root.mainloop()
