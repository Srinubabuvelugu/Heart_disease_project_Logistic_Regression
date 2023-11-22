from flask import Flask, render_template,request,jsonify
from src.pipeline.training_pipeline import *
from src.pipeline.prediction_pipeline import *

application = Flask(__name__)
app = application

## creating the route for home page
@app.route('/')
def homepage():
    return render_template('index.html')

## Creating route for Prediction 
@app.route('/predict',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('form.html')
                   
    else:
        data = CustomData(
            male = int(request.form.get('male')),
            age = int(request.form.get('age')),
            prevalentHyp = int(request.form.get('prevalentHyp')),
            totChol = float(request.form.get('totChol')),
            sysBP = float(request.form.get('sysBP')),
            diaBP = float(request.form.get('diaBP')),
            BMI = float(request.form.get('BMI')),
            heartRate = float(request.form.get('heartRate')),
            glucose = float(request.form.get('glucose'))
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)
        return render_template('result.html',final_result = results)


if __name__ == "__main__":
      app.run(host="0.0.0.0",debug=True)