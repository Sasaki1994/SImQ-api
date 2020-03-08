from flask import Flask
from flask import request
from flask import jsonify
from models.question_ai import QuestionAI

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

model = None


def load_model():
    return QuestionAI.load("models/2020-03-08_model.pickle")


@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }
    # ensure an feature was properly uploaded to our endpoint
    if request.method == "POST":
        req_json = request.get_json()
        if req_json.get("curriculum_id") and req_json.get("text"):
            # read feature from json
            curriculum_id = req_json.get("curriculum_id")
            text = req_json.get("text")

            # predicting
            res_texts = model.predict(curriculum_id, text)

            response["n_item"] = len(res_texts)
            response["text"] = res_texts

            # indicate that the request was a success
            response["success"] = True

    # return the data dictionary as a JSON response
    return jsonify(response)


@app.route("/", methods=["GET"])
def dep_check():
    return "Deploy is Success"


if __name__ == '__main__':
    print("**loading AI models...**")
    model = load_model()
    print("**Success**")
    app.run()
