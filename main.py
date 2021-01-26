from model_files import config
from flask import Flask, request, jsonify
from model_files.predictions import predict_repeat_contact
import joblib, os
from healthcheck import HealthCheck
# import google.cloud.logging
import logging
import traceback
from multiprocessing import Value

# client =google.cloud.logging.Client()
# client.setup_logging()

# Count how many predictions are being served with an auto increment counter
counter = Value('i', 0)



app = Flask("repeat_customer_visits")

# Load the Model 
model_object = os.path.join("./model_files", config.MODEL_PATH)
# Load the feature engineered pipeline containing numerical and categorical transformations
features_object = os.path.join("./model_files",config.FEATURE_TRANSFORMATION_OBJECT_PATH)

features_transform_pipe = joblib.load(features_object)
model = joblib.load(model_object)

# Creating a healthcheck url to monitor the health check and heartbeat of the application
health = HealthCheck(app, "/hcheck")

def healthCheckApp():
    """
    Function to check the health of the application
    This function can be customized in production to check whether the database is being served,
    monitor cpu and memory resource of flask app. We can ping this dummy url to ensure the app is
    up and running


    Returns: A boolean indicating if application is up or down
    """
    return True, "I am alive"

health.add_check(healthCheckApp)


@app.route('/', methods= ['POST'])
def predict():
    """
    Function to serve the predictions of repeat contacts classifier

    Returns:  a json response with Predictions and the score
    """
    if model:

        try:
            incoming_data = request.get_json()
            client_ip = request.environ['REMOTE_ADDR']
            # Keep only the variables contribution to model prediction
            repeat_contact = {key: [value] for key, value in incoming_data.items() if key.lower() not in config.NOT_TO_READ}
            
            with counter.get_lock():
                counter.value += 1
                out = counter.value
            predictions = predict_repeat_contact(repeat_contact, model, features_transform_pipe)
            app.logger.info(f"The prediction has been served for request id {counter} with client ip {client_ip}")
            
            # we can store the incoming_data and final predictions in the database 

            return jsonify(predictions)
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ("No model loaded")


