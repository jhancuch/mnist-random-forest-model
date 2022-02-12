from flask import Flask, request
import pickle
from numpy import array2string
import os

os.chdir('/mnt/c/Users/jwnha/Documents/_School/MSDS/2022 Winter/Machine Learning/Week 6/mnist-random-forest-model/model')

# define the path to the pickled model
model_path = "rf.pkl"
# unpickle the random forest model
with open(model_path, "rb") as file:
    unpickled_rf = pickle.load(file)

# define the app
app = Flask(__name__)

# use decorator to define the /score input method and define the predict function
@app.route("/score", methods=["POST", "GET"])
def predict_species():
    # create list and append inputs
    mnist_vector = []
    flower.append(request.args.get("mnist_vector"))

    # return the prediction
    return array2string(unpickled_rf.predict([digits]))


# run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port="5001")