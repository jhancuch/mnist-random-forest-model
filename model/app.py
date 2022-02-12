from flask import Flask, request
import pickle
from numpy import array2string
import os

os.chdir('/mnt/c/Users/jwnha/Documents/_School/MSDS/2022 Winter/Machine Learning/Week 6/mnist-random-forest-model/model')

model_path = "rf.pkl"

with open(model_path, "rb") as file:
    unpickled_rf = pickle.load(file)

# define the app
app = Flask(__name__)

# use decorator to define the /score input method and define the predict function
#@app.route("/score", methods=["POST", "GET"])
@app.route("/")
def predict_digit():
    # create list and append inputs
    mnist_vector = []
    mnist_vector.append(request.args.get("mnist_vector_1"))
    mnist_vector.append(request.args.get("mnist_vector_2"))
    mnist_vector.append(request.args.get("mnist_vector_3"))
    mnist_vector.append(request.args.get("mnist_vector_4"))
    mnist_vector.append(request.args.get("mnist_vector_5"))
    mnist_vector.append(request.args.get("mnist_vector_6"))
    mnist_vector.append(request.args.get("mnist_vector_7"))
    mnist_vector.append(request.args.get("mnist_vector_8"))
    mnist_vector.append(request.args.get("mnist_vector_9"))
    mnist_vector.append(request.args.get("mnist_vector_10"))
    mnist_vector.append(request.args.get("mnist_vector_11"))
    mnist_vector.append(request.args.get("mnist_vector_12"))
    mnist_vector.append(request.args.get("mnist_vector_13"))
    mnist_vector.append(request.args.get("mnist_vector_14"))
    mnist_vector.append(request.args.get("mnist_vector_15"))
    mnist_vector.append(request.args.get("mnist_vector_16"))
    mnist_vector.append(request.args.get("mnist_vector_17"))
    mnist_vector.append(request.args.get("mnist_vector_18"))
    mnist_vector.append(request.args.get("mnist_vector_19"))
    mnist_vector.append(request.args.get("mnist_vector_20"))
    mnist_vector.append(request.args.get("mnist_vector_21"))
    mnist_vector.append(request.args.get("mnist_vector_22"))
    mnist_vector.append(request.args.get("mnist_vector_23"))
    mnist_vector.append(request.args.get("mnist_vector_24"))
    mnist_vector.append(request.args.get("mnist_vector_25"))
    mnist_vector.append(request.args.get("mnist_vector_26"))
    mnist_vector.append(request.args.get("mnist_vector_27"))
    mnist_vector.append(request.args.get("mnist_vector_28"))
    mnist_vector.append(request.args.get("mnist_vector_29"))
    mnist_vector.append(request.args.get("mnist_vector_30"))
    mnist_vector.append(request.args.get("mnist_vector_31"))
    mnist_vector.append(request.args.get("mnist_vector_32"))
    mnist_vector.append(request.args.get("mnist_vector_33"))
    mnist_vector.append(request.args.get("mnist_vector_34"))
    mnist_vector.append(request.args.get("mnist_vector_35"))
    mnist_vector.append(request.args.get("mnist_vector_36"))
    mnist_vector.append(request.args.get("mnist_vector_37"))
    mnist_vector.append(request.args.get("mnist_vector_38"))
    mnist_vector.append(request.args.get("mnist_vector_39"))
    mnist_vector.append(request.args.get("mnist_vector_40"))
    mnist_vector.append(request.args.get("mnist_vector_41"))
    mnist_vector.append(request.args.get("mnist_vector_42"))
    mnist_vector.append(request.args.get("mnist_vector_43"))
    mnist_vector.append(request.args.get("mnist_vector_44"))
    mnist_vector.append(request.args.get("mnist_vector_45"))
    mnist_vector.append(request.args.get("mnist_vector_46"))
    mnist_vector.append(request.args.get("mnist_vector_47"))
    mnist_vector.append(request.args.get("mnist_vector_48"))
    mnist_vector.append(request.args.get("mnist_vector_49"))
    mnist_vector.append(request.args.get("mnist_vector_50"))
    mnist_vector.append(request.args.get("mnist_vector_51"))
    mnist_vector.append(request.args.get("mnist_vector_52"))
    mnist_vector.append(request.args.get("mnist_vector_53"))
    mnist_vector.append(request.args.get("mnist_vector_54"))
    mnist_vector.append(request.args.get("mnist_vector_55"))
    mnist_vector.append(request.args.get("mnist_vector_56"))
    mnist_vector.append(request.args.get("mnist_vector_57"))
    mnist_vector.append(request.args.get("mnist_vector_58"))
    mnist_vector.append(request.args.get("mnist_vector_59"))
    mnist_vector.append(request.args.get("mnist_vector_60"))
    mnist_vector.append(request.args.get("mnist_vector_61"))
    mnist_vector.append(request.args.get("mnist_vector_62"))
    mnist_vector.append(request.args.get("mnist_vector_63"))
    mnist_vector.append(request.args.get("mnist_vector_64"))

    # return the prediction
    return array2string(unpickled_rf.predict([mnist_vector]))


# run the app
if __name__ == "__main__":
    app.run()
