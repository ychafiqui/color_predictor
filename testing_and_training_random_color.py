import tensorflow as tf
import turtle as t
from random import randrange
import pickle

class testtt:
    def __init__(self, color):
        self.color = color
    def pre(self, prefiction):
        self.prediction = prefiction

# getting the dataset with pickle
with open('dataset.pkl', 'rb') as f:
    x_train, y_train = pickle.load(f)

# loading the model
model = tf.keras.models.load_model("color_predictor.h5")

# predict function
def predict_color(color):
    prediction = model.predict([color])
    if prediction < 0.5: black_guess()
    else: white_guess()

# random color generator
def random_color():
    red = randrange(0, 256) / 255
    green = randrange(0, 256) / 255
    blue = randrange(0, 256) / 255
    return red, green, blue

# generating the first color
color = random_color()
testt = testtt(color)

wn = t.Screen()
wn.title("Color predictor model testing")
wn.bgcolor(color)
wn.setup(width=800, height=600)
wn.tracer(0)

# text sample for color preview
text_sample = "Text"

text = t.Turtle()
text.speed(0)
text.penup()
text.hideturtle()
text.goto(0, 0)

instructions_text = "Up arrow: Up-vote the AI prediction"
instructions_text += "\nDown arrow: Down-vote the AI prediction"
instructions_text += "\nSpace bar: Re-train model"
instructions_text += "\nEsc: Cancel and quit"

instructions = t.Turtle()
instructions.speed(0)
instructions.penup()
instructions.hideturtle()
instructions.goto(0, 230)

def black_guess():
    testt.pre(0)
    text.clear()
    text.color("black")
    text.write(text_sample, align="center", font=("Courier", 25, "bold"))
    instructions.clear()
    instructions.color("black")
    instructions.write(instructions_text, align="center", font=("Courier", 10, "bold"))

def white_guess():
    testt.pre(1)
    text.clear()
    text.color("white")
    text.write(text_sample, align="center", font=("Courier", 25, "bold"))
    instructions.clear()
    instructions.color("white")
    instructions.write(instructions_text, align="center", font=("Courier", 10, "bold"))

predict_color(color)

def up_vote():
    # generating a new color
    testt.color = random_color()
    wn.bgcolor(testt.color)
    predict_color(testt.color)

def down_vote():
    # adding the new color to the dataset
    x_train.append(testt.color)
    if testt.prediction == 1: y_train.append(0)
    else: y_train.append(1)
    # generating a new color
    testt.color = random_color()
    wn.bgcolor(testt.color)
    predict_color(testt.color)

def start_training():
    # testing if the dataset is empty
    if len(x_train) > 0:
        with open('dataset.pkl', 'wb') as f:
            pickle.dump([x_train, y_train], f) # saving the new dataset with pickle
        wn.bye() # closing the window
        model.fit(x_train, y_train, epochs=30) # model training
        model.save("color_predictor.h5") # saving the model
    else: print("Dataset is empty, you need to chose at least one color!")

def quit_program():
    wn.bye()

wn.listen()
wn.onkeypress(up_vote, "Up")
wn.onkeypress(down_vote, "Down")
wn.onkeypress(start_training, "Return")
wn.onkeypress(start_training, "space")
wn.onkeypress(quit_program, "Escape")

wn.mainloop()