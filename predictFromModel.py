import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import pickle


class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)

            data = preprocessor.dropUnnecessaryColumns(data, ['id', 'region', 'url', 'region_url', 'image_url', 'state', 'type', 'dogs_allowed'])

            # get encoded values for categorical data

            data = preprocessor.encodeCategoricalValuesPrediction(data)

            data_scaled = pandas.DataFrame(preprocessor.standardScalingData(data), columns=data.columns)

            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')

            ##Code changed
            clusters=kmeans.predict(data_scaled)#drops the first column for cluster prediction
            data_scaled['clusters']=clusters
            clusters=data_scaled['clusters'].unique()
            result=[] # initialize balnk list for storing predicitons
            for i in clusters:
                cluster_data= data_scaled[data_scaled['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                for val in (model.predict(cluster_data.values)):
                    result.append(val)
            result = pandas.DataFrame(result,columns=['Prediction'])
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True,mode='a+') #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path, result.head().to_json(orient="records")




