from model_files import config
from flask import Flask, request, jsonify
from model_files.predictions import predict_repeat_contact
import joblib, os
from healthcheck import HealthCheck
# import google.cloud.logging
import logging
from multiprocessing import Value

# client =google.cloud.logging.Client()
# client.setup_logging()

# Count how many predictions are being served with an auto increment counter
counter = Value('i', 0)



app = Flask("repeat_customer_visits")

# @app.route('/', methods= ['GET'])
# def ping():
#     return "Pinging model application!!"

# logging.basicConfig(filename = "repeat_customers_visit_app.log", level = logging.INFO,
#                     format = '%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

model_object = os.path.join("./model_files", config.MODEL_PATH)
features_object = os.path.join("./model_files",config.FEATURE_TRANSFORMATION_OBJECT_PATH)

features_transform_pipe = joblib.load(features_object)
model = joblib.load(model_object)

health = HealthCheck(app, "/hcheck")

def healthCheckApp():
    return True, "I am alive"

health.add_check(healthCheckApp)


@app.route('/predict', methods= ['POST'])
def predict():

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


# if __name__ == '__main__':
#     app.run( debug = True, port = 9096)

if __name__ == '__main__':
    app.run( host = '0.0.0.0', port = 9096)

