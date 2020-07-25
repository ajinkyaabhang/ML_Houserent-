import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import pickle
from imblearn.over_sampling import RandomOverSampler
from sklearn_pandas import CategoricalImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: Ajinkya Abhang
        Version: 1.0
        Revisions: None

        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

                Written By: Ajinkya Abhang
                Version: 1.0
                Revisions: None

        """
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                        Written By: Ajinkya Abhang
                        Version: 1.0
                        Revisions: None

                """
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumns(self,data,columnNameList):
        """
                        Method Name: is_null_present
                        Description: This method drops the unwanted columns as discussed in EDA section.

                        Written By: Ajinkya Abhang
                        Version: 1.0
                        Revisions: None

                                """
        data = data.drop(columnNameList,axis=1)
        return data

    def dropOutliers(self,data):
        """
                        Method Name: dropOutliers
                        Description: This method drops the outliers as discussed in EDA section.

                        Written By: Ajinkya Abhang
                        Version: 1.0
                        Revisions: None

                                """
        self.logger_object.log(self.file_object, 'Entered the dropOutliers method of the Preprocessor class')
        try:
            data.drop(data[(data['price'] > 4000)].index, inplace=True)
            data.drop(data[(data['sqfeet'] > 4000)].index, inplace=True)
            data.drop(data[np.abs(data.beds - data.beds.mean()) >= (3 * data.beds.std())].index, inplace=True)
            data.drop(data[np.abs(data.baths - data.baths.mean()) >= (3 * data.baths.std())].index, inplace=True)
            data.drop(data[(data['price'] < 400)].index, inplace=True)
            data.drop(data[(data['sqfeet'] < 100)].index, inplace=True)
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in dropOutliers method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'dropOutliers Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropOutliers_predict(self,data):
        """
                        Method Name: dropOutliers_predict
                        Description: This method drops the outliers as discussed in EDA section.

                        Written By: Ajinkya Abhang
                        Version: 1.0
                        Revisions: None

                                """
        self.logger_object.log(self.file_object, 'Entered the dropOutliers_predict method of the Preprocessor class')
        try:
            data.drop(data[(data['sqfeet'] > 4000)].index, inplace=True)
            data.drop(data[np.abs(data.beds - data.beds.mean()) >= (3 * data.beds.std())].index, inplace=True)
            data.drop(data[np.abs(data.baths - data.baths.mean()) >= (3 * data.baths.std())].index, inplace=True)
            data.drop(data[(data['sqfeet'] < 100)].index, inplace=True)
            return data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in dropOutliers_predict method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'dropOutliers_predict Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def replaceInvalidValuesWithNull(self,data):

        """
                               Method Name: is_null_present
                               Description: This method replaces invalid values i.e. '?' with null, as discussed in EDA.

                               Written By: Ajinkya Abhang
                               Version: 1.0
                               Revisions: None

                                       """

        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def is_null_present(self,data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                                On Failure: Raise Exception

                                Written By: Ajinkya Abhang
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in self.null_counts:
                if i>0:
                    self.null_present=True
                    break
            if(self.null_present): # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            self.logger_object.log(self.file_object,'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self,data):
        """
                                                Method Name: encodeCategoricalValues
                                                Description: This method encodes all the categorical values in the training set.
                                                Output: A Dataframe which has all the categorical values encoded.
                                                On Failure: Raise Exception

                                                Written By: Ajinkya Abhang
                                                Version: 1.0
                                                Revisions: None
        """


    # We can impute the categorical values like below:
        features_nan = [feature for feature in data.columns if data[feature].isnull().sum() > 0 and data[feature].dtypes == 'O']

        imputer = CategoricalImputer()

        if len(features_nan) != 0:
            for cat_feature in features_nan:
                data[cat_feature] = imputer.fit_transform(data[cat_feature])

        # We can impute the non-categorical values like below:
        numerical_with_nan = [feature for feature in data.columns if
                            data[feature].isnull().sum() > 1 and data[feature].dtypes != 'O']

        if len(numerical_with_nan) != 0:
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            data[numerical_with_nan] = imputer.fit_transform(data[numerical_with_nan])


        # We can use label encoder for encoding
        labelencoder = LabelEncoder()
        dummy_features = ['laundry_options', 'parking_options']

        for feature in dummy_features:
            data[feature] = labelencoder.fit_transform(data[feature])

        for feature in dummy_features:
            data_df = pd.get_dummies(data, columns=['laundry_options', 'parking_options'], drop_first=True)



        return data_df


    def encodeCategoricalValuesPrediction(self,data):
        """
                                               Method Name: encodeCategoricalValuesPrediction
                                               Description: This method encodes all the categorical values in the prediction set.
                                               Output: A Dataframe which has all the categorical values encoded.
                                               On Failure: Raise Exception

                                               Written By: Ajinkya Abhang
                                               Version: 1.0
                                               Revisions: None
                            """

        # We can impute the categorical values like below:
        features_nan = [feature for feature in data.columns if
                        data[feature].isnull().sum() > 0 and data[feature].dtypes == 'O']

        imputer = CategoricalImputer()

        if len(features_nan) != 0:
            for cat_feature in features_nan:
                data[cat_feature] = imputer.fit_transform(data[cat_feature])

        # We can impute the non-categorical values like below:
        numerical_with_nan = [feature for feature in data.columns if
                              data[feature].isnull().sum() > 1 and data[feature].dtypes != 'O']

        if len(numerical_with_nan) != 0:
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
            data[numerical_with_nan] = imputer.fit_transform(data[numerical_with_nan])

        # We can use label encoder for encoding
        df_new = pd.DataFrame({
            'laundry_options_1': [np.nan] * data.shape[0],
            'laundry_options_2': [np.nan] * data.shape[0],
            'laundry_options_3': [np.nan] * data.shape[0],
            'laundry_options_4': [np.nan] * data.shape[0],
            'parking_options_1': [np.nan] * data.shape[0],
            'parking_options_2': [np.nan] * data.shape[0],
            'parking_options_3': [np.nan] * data.shape[0],
            'parking_options_4': [np.nan] * data.shape[0],
            'parking_options_5': [np.nan] * data.shape[0],
            'parking_options_6': [np.nan] * data.shape[0]
        })

        dat = pd.concat([data, df_new], axis=1)

        for i in range(data.shape[0]):
            if (dat['laundry_options'][i] == 'w/d in unit'):
                dat['laundry_options_1'][i] = 0
                dat['laundry_options_2'][i] = 0
                dat['laundry_options_3'][i] = 0
                dat['laundry_options_4'][i] = 1
            elif (dat['laundry_options'][i] == 'w/d hookups'):
                dat['laundry_options_1'][i] = 0
                dat['laundry_options_2'][i] = 0
                dat['laundry_options_3'][i] = 1
                dat['laundry_options_4'][i] = 0
            elif (dat['laundry_options'][i] == 'laundry on site'):
                dat['laundry_options_1'][i] = 1
                dat['laundry_options_2'][i] = 0
                dat['laundry_options_3'][i] = 0
                dat['laundry_options_4'][i] = 0
            elif (dat['laundry_options'][i] == 'no laundry on site'):
                dat['laundry_options_1'][i] = 0
                dat['laundry_options_2'][i] = 1
                dat['laundry_options_3'][i] = 0
                dat['laundry_options_4'][i] = 0
            elif (dat['laundry_options'][i] == 'laundry in bldg'):
                dat['laundry_options_1'][i] = 0
                dat['laundry_options_2'][i] = 0
                dat['laundry_options_3'][i] = 0
                dat['laundry_options_4'][i] = 0

        for i in range(data.shape[0]):
            if (dat['parking_options'][i] == 'carport'):
                dat['parking_options_1'][i] = 1
                dat['parking_options_2'][i] = 0
                dat['parking_options_3'][i] = 0
                dat['parking_options_4'][i] = 0
                dat['parking_options_5'][i] = 0
                dat['parking_options_6'][i] = 0
            elif (dat['parking_options'][i] == 'detached garage'):
                dat['parking_options_1'][i] = 0
                dat['parking_options_2'][i] = 1
                dat['parking_options_3'][i] = 0
                dat['parking_options_4'][i] = 0
                dat['parking_options_5'][i] = 0
                dat['parking_options_6'][i] = 0
            elif (dat['parking_options'][i] == 'no parking'):
                dat['parking_options_1'][i] = 0
                dat['parking_options_2'][i] = 0
                dat['parking_options_3'][i] = 1
                dat['parking_options_4'][i] = 0
                dat['parking_options_5'][i] = 0
                dat['parking_options_6'][i] = 0
            elif (dat['parking_options'][i] == 'off-street parking'):
                dat['parking_options_1'][i] = 0
                dat['parking_options_2'][i] = 0
                dat['parking_options_3'][i] = 0
                dat['parking_options_4'][i] = 1
                dat['parking_options_5'][i] = 0
                dat['parking_options_6'][i] = 0
            elif (dat['parking_options'][i] == 'street parking'):
                dat['parking_options_1'][i] = 0
                dat['parking_options_2'][i] = 0
                dat['parking_options_3'][i] = 0
                dat['parking_options_4'][i] = 0
                dat['parking_options_5'][i] = 1
                dat['parking_options_6'][i] = 0
            elif (dat['parking_options'][i] == 'valet parking'):
                dat['parking_options_1'][i] = 0
                dat['parking_options_2'][i] = 0
                dat['parking_options_3'][i] = 0
                dat['parking_options_4'][i] = 0
                dat['parking_options_5'][i] = 0
                dat['parking_options_6'][i] = 1
            elif (dat['parking_options'][i] == 'attached garage'):
                dat['parking_options_1'][i] = 0
                dat['parking_options_2'][i] = 0
                dat['parking_options_3'][i] = 0
                dat['parking_options_4'][i] = 0
                dat['parking_options_5'][i] = 0
                dat['parking_options_6'][i] = 0

        dat.drop(['laundry_options', 'parking_options'], axis=1, inplace = True)

        return dat

    def handleImbalanceDataset(self,X,Y):
        """
                                                      Method Name: handleImbalanceDataset
                                                      Description: This method handles the imbalance in the dataset by oversampling.
                                                      Output: A Dataframe which is balanced now.
                                                      On Failure: Raise Exception

                                                      Written By: Ajinkya Abhang
                                                      Version: 1.0
                                                      Revisions: None
                                   """



        rdsmple = RandomOverSampler()
        x_sampled, y_sampled = rdsmple.fit_sample(X, Y)

        return x_sampled,y_sampled

    def standardScalingData(self, X):

        scalar = StandardScaler()
        X_scaled = scalar.fit_transform(X)

        return X_scaled

    def impute_missing_values(self, data):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception

                                        Written By: Ajinkya Abhang
                                        Version: 1.0
                                        Revisions: None
                     """
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            # rounding the value because KNNimputer returns value between 0 and 1, but we need either 0 or 1
            self.new_data=pd.DataFrame(data=np.round(self.new_array), columns=self.data.columns)
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                                                Written By: Ajinkya Abhang
                                                Version: 1.0
                                                Revisions: None
                             """
        self.logger_object.log(self.file_object, 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # check if standard deviation is zero
                    self.col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()