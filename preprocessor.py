from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
class Train_preprocessor():
    def __init__(self, json_file,binary=False):
            """
            Inicializa una instancia de Train_preprocessor.
            Abre y carga un archivo JSON para mapear las columnas y establece el atributo Tree.
            :param json_file: Ruta del archivo JSON para mapeo de columnas.
            :param Tree: Valor booleano, por defecto True.
            """
            self.binary=binary
            with open(json_file, 'r') as file:  # Abre el archivo JSON
                self.jsonmapping = json.load(file)  # Carga el contenido del archivo JSON
#--------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------
    def clasif_diag(self,diag):
        try:
            num = float(diag)
            if 0<= num <= 139:
                return 'INFECTIOUS AND PARASITIC DISEASES'
            elif 140 <= num <= 239:
                return 'NEOPLASMS'
            elif 240 <= num <= 279:
                return 'ENDOCRINE, NUTRITIONAL AND METABOLIC DISEASES, AND IMMUNITY DISORDERS'
            elif 280 <= num <= 289:
                return 'DISEASES OF THE BLOOD AND BLOOD-FORMING ORGANS'
            elif 290 <= num <= 319:
                return 'MENTAL DISORDERS'
            elif 320 <= num <= 389:
                return 'DISEASES OF THE NERVOUS SYSTEM AND SENSE ORGANS'
            elif 390 <= num <= 459:
                return 'DISEASES OF THE CIRCULATORY SYSTEM'
            elif 460 <= num <= 519:
                return 'DISEASES OF THE RESPIRATORY SYSTEM'
            elif 520 <= num <= 579:
                return 'DISEASES OF THE DIGESTIVE SYSTEM'
            elif 580 <= num <= 629:
                return 'DISEASES OF THE GENITOURINARY SYSTEM'
            elif 630 <= num <= 679:
                return 'COMPLICATIONS OF PREGNANCY, CHILDBIRTH, AND THE PUERPERIUM'
            elif 680 <= num <= 709:
                return 'DISEASES OF THE SKIN AND SUBCUTANEOUS TISSUE'
            elif 710 <= num <= 739:
                return 'DISEASES OF THE MUSCULOSKELETAL SYSTEM AND CONNECTIVE TISSUE'
            elif 740 <= num <= 759:
                return 'CONGENITAL ANOMALIES'
            elif 760 <= num <= 779:
                return 'CERTAIN CONDITIONS ORIGINATING IN THE PERINATAL PERIOD'
            elif 780 <= num <= 799:
                return 'SYMPTOMS, SIGNS, AND ILL-DEFINED CONDITIONS'
            elif 800 <= num <= 999:
                return 'INJURY AND POISONING'

        except ValueError:
            num = str(diag)
            if num.startswith('E'):
                return 'SUPPLEMENTARY CLASSIFICATION OF FACTORS INFLUENCING HEALTH STATUS AND CONTACT WITH HEALTH SERVICES'
            elif num.startswith('V'):
                return 'SUPPLEMENTARY CLASSIFICATION OF EXTERNAL CAUSES OF INJURY AND POISONING'
#--------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------
    def ColumnMapper(self, dataset, variables):
        """
        Mapea los valores de varias columnas en un conjunto de datos.
        :param dataset: Conjunto de datos a modificar.
        :param variables: Lista de nombres de columnas en el conjunto de datos.
        :return: Conjunto de datos con las columnas mapeadas.
        """
        for variable in variables:
            if variable in self.jsonmapping:
                if variable!='age':
                    dataset[variable]=dataset[variable].astype(str)
                dataset[variable] = dataset[variable].map(self.jsonmapping[variable])
                dataset = dataset.drop(dataset.loc[(dataset[variable]=='expired') | (dataset[variable]=='NULL')].index)
            else:
                print(f"Advertencia: no se encontró mapeo para la variable '{variable}'")
        return dataset
#--------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------- COLUMNS TRANSFORMATION -------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------    
    def medication_varibale(self,df):
        df['medication/day']=df['num_medications']/df['time_in_hospital']
        return df
#--------------------------------------------------------------------------------------------------------------------------------------------------    
    def number_service_uses(self,df):
        #Creacion de una variable de uso de servicios sanitaarios
        df['n_service_uses'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        df = df.drop(df[['number_outpatient', 'number_emergency', 'number_inpatient']],axis=1)
        df['sqrt_n_service_uses'] = np.sqrt(df['n_service_uses'])
        return df
#--------------------------------------------------------------------------------------------------------------------------------------------------
    def variable_configuration(self,df):
        #Eliminar los pacientes que estan repetidos
        df=df.drop_duplicates(subset=['patient_nbr'],keep='first')
        #Eliminación de IDs
        df = df.drop(['payer_code','encounter_id','patient_nbr'], axis = 1)
        #Seleccion de casos
        df0 = df[df['readmitted']=='NO']
        df1 = df[df['readmitted']=='>30']
        df2 = df[df['readmitted']=='<30']

        df = pd.concat([df0, df1, df2])
        #Transformacion de categorico a numerico

        d = {'Female': 0, 'Male': 1,'Unknown/Invalid':-1}
        df['gender'] = df['gender'].map(d).astype(int)
        df=df[df['gender']!=-1]

        return df

    def mapping_to_num_diag(self,df):
        df['diag_1'] = df['diag_1'].apply(self.clasif_diag)
        df['diag_2'] = df['diag_2'].apply(self.clasif_diag)
        df['diag_3'] = df['diag_3'].apply(self.clasif_diag)


        d = {'INFECTIOUS AND PARASITIC DISEASES':0,
            'NEOPLASMS':1,
            'ENDOCRINE, NUTRITIONAL AND METABOLIC DISEASES, AND IMMUNITY DISORDERS':2,
            'DISEASES OF THE BLOOD AND BLOOD-FORMING ORGANS':3,
            'MENTAL DISORDERS':4,
            'DISEASES OF THE NERVOUS SYSTEM AND SENSE ORGANS':5,
            'DISEASES OF THE CIRCULATORY SYSTEM':6,
            'DISEASES OF THE RESPIRATORY SYSTEM':7,
            'DISEASES OF THE DIGESTIVE SYSTEM':8,
            'DISEASES OF THE GENITOURINARY SYSTEM':9,
            'COMPLICATIONS OF PREGNANCY, CHILDBIRTH, AND THE PUERPERIUM':10,
            'DISEASES OF THE SKIN AND SUBCUTANEOUS TISSUE':11,
            'DISEASES OF THE MUSCULOSKELETAL SYSTEM AND CONNECTIVE TISSUE':12,
            'CONGENITAL ANOMALIES':13,
            'CERTAIN CONDITIONS ORIGINATING IN THE PERINATAL PERIOD':14,
            'SYMPTOMS, SIGNS, AND ILL-DEFINED CONDITIONS':15,
            'INJURY AND POISONING' :16,
            'SUPPLEMENTARY CLASSIFICATION OF FACTORS INFLUENCING HEALTH STATUS AND CONTACT WITH HEALTH SERVICES' :17,
            'SUPPLEMENTARY CLASSIFICATION OF EXTERNAL CAUSES OF INJURY AND POISONING':18}
        df=df.dropna(subset=['diag_1','diag_2','diag_3'])
        df['diag_1'] = df['diag_1'].map(d).astype(int)
        df['diag_2'] = df['diag_2'].map(d).astype(int)
        df['diag_3'] = df['diag_3'].map(d).astype(int)
        return df
    def mapping_target_var(self,df):
        if self.binary:
            d={
                '<30':1,
                '>30':1,
                'NO':0,
            }
        else:
            d={
                '<30':2,
                '>30':1,
                'NO':0,
            }
        df['readmitted']=df['readmitted'].map(d).astype(int)
        return df
#--------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------PREPROCESSING ALL-------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------
    def transform_eda(self,df):
        df=self.ColumnMapper(df,["age","admission_source_id","discharge_disposition_id","admission_type_id",'race','metformin','insulin','diabetesMed','change'])
        if df.count()[0]==0:
            print(0)
        df=self.variable_configuration(df)
        if df.count()[0]==0:
            print(1)
        df=self.number_service_uses(df)
        if df.count()[0]==0:
            print(2)
        df=self.medication_varibale(df)
        if df.count()[0]==0:
            print('REVISA MEDICATION_VARIABLE_MAPPING')
        df=self.mapping_to_num_diag(df)
        if df.count()[0]==0:
            print('OJO A LA VAIABLE DIAG: NO SE MAPEA BIEN->algo ha fallado en el mapeo del target_var')
        df=self.mapping_target_var(df)
        if df.count()[0]==0:
            print('algo ha fallado en el mapeo del target_var')

        return df
    
    
    
    
    def transform(self,df):
        df=self.ColumnMapper(df,["age","admission_source_id","discharge_disposition_id","admission_type_id",'race','metformin','insulin','diabetesMed','change'])
        if df.count()[0]==0:
            print(0)
        df=self.variable_configuration(df)
        if df.count()[0]==0:
            print(1)
        df=self.number_service_uses(df)
        if df.count()[0]==0:
            print(2)
        df=self.medication_varibale(df)
        if df.count()[0]==0:
            print('REVISA MEDICATION_VARIABLE_MAPPING')
        df=self.mapping_to_num_diag(df)
        if df.count()[0]==0:
            print('OJO A LA VAIABLE DIAG: NO SE MAPEA BIEN->algo ha fallado en el mapeo del target_var')
        df=self.mapping_target_var(df)
        if df.count()[0]==0:
            print('algo ha fallado en el mapeo del target_var')
        final_df=pd.DataFrame()
        for i in self.jsonmapping["mandatory_variables"].keys():
            final_df[i]=df[i]
        final_df = pd.get_dummies(final_df, columns = ['discharge_disposition_id','admission_type_id'])    
        return final_df



#--------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------------------------------------------------
class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, json_file, categorical_features_ohe, categorical_features_LE,numerical_features):
        self.json_file = json_file
        self.categorical_features_to_ohe = categorical_features_ohe
        self.categorical_features_to_LE = categorical_features_LE
        self.numerical_features = numerical_features
        self.custom_preprocessor = Train_preprocessor(self.json_file)
        
        self.OHE_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        self.LE_transformer = Pipeline(steps=[
            ('LabelEncoder', LabelEncoder(handle_unknown='ignore'))])

        self.numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, self.numerical_features),
                ('ohe', self.OHE_transformer, self.categorical_features_to_ohe),
                ('le', self.LE_transformer, self.categorical_features_to_LE)])
        
        self.pipeline = Pipeline(steps=[('custom_preprocessor', self.custom_preprocessor),
                                        ('preprocessor', self.preprocessor)])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)