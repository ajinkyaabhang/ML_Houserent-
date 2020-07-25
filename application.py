from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
import json
import pickle
from file_operations import file_methods
from application_logging import logger
from sklearn.preprocessing import StandardScaler

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__)
app = application
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict_json", methods=['POST'])
@cross_origin()
def predictJsonRouteClient():
    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path)  # object initialization

            # predicting for dataset present in database
            path, json_predictions = pred.predictionFromModel()
            return Response("Prediction File created at !!!" + str(path) + ' and few of the predictions are ' + str(json.loads(json_predictions)))
        else:
           print('Nothing Matched')

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        path = request.form['Default_File_Predict']

        pred_val = pred_validation(path)  # object initialization

        pred_val.prediction_validation()  # calling the prediction_validation function

        pred = prediction(path)  # object initialization

        # predicting for dataset present in database
        path, json_predictions = pred.predictionFromModel()

        return render_template('results.html',prediction='Prediction has been saved at {} and few of the predictions are '.format(path) +' ' + str(json.loads(json_predictions)))

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/predict_new", methods=['POST'])
@cross_origin()
def predictNewRouteClient():
    try:
        sqfeet=int(request.form['sqfeet'])
        beds = int(request.form['beds'])
        baths = float(request.form['baths'])
        is_cats_allowed = request.form['cats_allowed']
        if (is_cats_allowed == 'Yes'):
            cats_allowed = 1
        else:
            cats_allowed = 0
        is_smoking_allowed = request.form['smoking_allowed']
        if (is_smoking_allowed == 'Yes'):
            smoking_allowed = 1
        else:
            smoking_allowed = 0
        is_wheelchair_access = request.form['wheelchair_access']
        if (is_wheelchair_access == 'Yes'):
            wheelchair_access = 1
        else:
            wheelchair_access = 0
        is_electric_vehicle_charge = request.form['electric_vehicle_charge']
        if (is_electric_vehicle_charge == 'Yes'):
            electric_vehicle_charge = 1
        else:
            electric_vehicle_charge = 0
        is_comes_furnished = request.form['comes_furnished']
        if (is_comes_furnished == 'Yes'):
            comes_furnished = 1
        else:
            comes_furnished = 0
        lat = float(request.form['lat'])
        long = float(request.form['long'])
        is_parking_options = request.form['parking_options']
        if (is_parking_options == 'carport'):
            parking_options_1 = 1
            parking_options_2 = 0
            parking_options_3 = 0
            parking_options_4 = 0
            parking_options_5 = 0
            parking_options_6 = 0
        elif (is_parking_options == 'detached garage'):
            parking_options_1 = 0
            parking_options_2 = 1
            parking_options_3 = 0
            parking_options_4 = 0
            parking_options_5 = 0
            parking_options_6 = 0
        elif (is_parking_options == 'no parking'):
            parking_options_1 = 0
            parking_options_2 = 0
            parking_options_3 = 1
            parking_options_4 = 0
            parking_options_5 = 0
            parking_options_6 = 0
        elif (is_parking_options == 'off-street parking'):
            parking_options_1 = 0
            parking_options_2 = 0
            parking_options_3 = 0
            parking_options_4 = 1
            parking_options_5 = 0
            parking_options_6 = 0
        elif (is_parking_options == 'street parking'):
            parking_options_1 = 0
            parking_options_2 = 0
            parking_options_3 = 0
            parking_options_4 = 0
            parking_options_5 = 1
            parking_options_6 = 0
        elif (is_parking_options == 'valet parking'):
            parking_options_1 = 0
            parking_options_2 = 0
            parking_options_3 = 0
            parking_options_4 = 0
            parking_options_5 = 0
            parking_options_6 = 1
        else:
            parking_options_1 = 0
            parking_options_2 = 0
            parking_options_3 = 0
            parking_options_4 = 0
            parking_options_5 = 0
            parking_options_6 = 0

        is_laundry_options = request.form['laundry_options']
        if (is_laundry_options == 'w/d in unit'):
            laundry_options_1 = 0
            laundry_options_2 = 0
            laundry_options_3 = 0
            laundry_options_4 = 1
        elif (is_laundry_options == 'w/d hookups'):
            laundry_options_1 = 0
            laundry_options_2 = 0
            laundry_options_3 = 1
            laundry_options_4 = 0
        elif (is_laundry_options == 'laundry on site'):
            laundry_options_1 = 1
            laundry_options_2 = 0
            laundry_options_3 = 0
            laundry_options_4 = 0
        elif (is_laundry_options == 'no laundry on site'):
            laundry_options_1 = 0
            laundry_options_2 = 1
            laundry_options_3 = 0
            laundry_options_4 = 0
        else:
            laundry_options_1 = 0
            laundry_options_2 = 0
            laundry_options_3 = 0
            laundry_options_4 = 0

        filename = "models/KMeans/KMeans.sav"
        loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
        # predictions using the loaded model file
        clusters=loaded_model.predict([[sqfeet, beds, baths, cats_allowed, smoking_allowed, wheelchair_access, electric_vehicle_charge, comes_furnished,
                                        lat, long, laundry_options_1, laundry_options_2, laundry_options_3, laundry_options_4, parking_options_1,
                                        parking_options_2, parking_options_3, parking_options_4, parking_options_5, parking_options_6]])
        file_object = open("Prediction_Logs/Prediction_Log_single.txt", 'a+')
        log_writer = logger.App_Logger()
        file_loader = file_methods.File_Operation(file_object, log_writer)

        model_name = file_loader.find_correct_model_file(clusters[0])
        model = file_loader.load_model(model_name)
        scalar = StandardScaler()
        X_scaled = scalar.fit_transform([[sqfeet, beds, baths, cats_allowed, smoking_allowed, wheelchair_access, electric_vehicle_charge, comes_furnished,
                                        lat, long, laundry_options_1, laundry_options_2, laundry_options_3, laundry_options_4, parking_options_1,
                                        parking_options_2, parking_options_3, parking_options_4, parking_options_5, parking_options_6]])
        result = model.predict(X_scaled)
        log_writer.log(file_object, 'End of Prediction')
        file_object.close()

        return render_template('results.html',prediction='Your House rent prediction is {} USD'.format(round(result[0], 2)))

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json['folderPath'] is not None:
            path = request.json['folderPath']
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function

            trainModelObj = trainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    #host = '0.0.0.0'
    #port = 5000
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()
    app.run(debug=True)