#Creates new data dictionary from dataset and replaces existing with new one
@analytics_toolsets.route('/featureStatistics',methods=['PUT'])
class generateFeatureStats(Resource):
    @api.response(*response_400)
    @api.expect(BaseSchema.data_dictionary_post)
    def put(self, model_meta_id, data_dictionary_version_id):
        """Creates a new data dictionary from a dataset or replaces the existing data dictionary with a given one"""
        data = request.json

        try:
            data_dict = DataDict(os_credentials=objectstore_credentials, os_container=data['os_container'],
            object_name=data['object_name'], object_version=data_dictionary_version_id)
            if data['input_format'] == 'file':
                dd = data_dict.generate_and_store(inp_form=data['input_format'], input_path=data['input_path'])
            else:
                dd = data_dict.generate_and_store(inp_form=data['input_format'], db_url=data['db_url'],
                    schema=data['schema'], tables=data['tables'], observation_ids=data['oids'])
        except Exception as e:
            raise exceptions.ParseError("Unable to generate Feature Statistics for this input")

        return 201



class CohortCharacterstics():
    def __init__(self, inp_form=None, input_path=None, db_url=None, schema=None, table=None, observation_ids=None, selection_criteria=None):
        # Access the dictionary from the object store
        columnList = observation_ids
        columnStr = ','.join(columnList)

        #output_log = open("./data_quality_check_log.txt", "w")
        print (columnList)

        print(columnStr)


        if inp_form == "file":
            file_name = os.path.basename(input_path)
            sc = SparkContext()
            sqlContext = SQLContext(sc)

            # read file
            input_fileformat = os.path.splitext(input_path)[1][1:].strip().lower()
            df = sqlContext.read.load(input_path, format=input_fileformat, inferSchema=True, header=True)

            print (df)
        else:
            engine = sqlalchemy.create_engine(db_url, echo=True)
            db_conn = engine.connect()

            query_str = ''' SELECT {0} 
                            FROM {1}.{2}'''.format(columnStr,schema,table)

            print (query_str)

            #result = pd.read_sql_query(query_str, db_conn)
            #print (result)

from analytics_toolsets import DataDict, DataQuality, CohortCharacterstics



@analytics_toolsets.route('/cohortStatistics',methods=['PUT'])
class generateCohortStats(Resource):
    resource_fields = api.model('Resource', {
        'input_format': fields.String(enum=["file", "db"]),
        'input_path': fields.String,
        'db_url': fields.String,
        'schema': fields.String,
        'table': fields.String,
        'oids': fields.List(fields.String),
        'selection_criteria': fields.List(fields.String)
    })
    @api.response(*response_400)
    @api.expect(resource_fields)
    def put(self):
        """Creates cohort characterstics from a dataset or replaces the existing cohort statistics with a given one"""
        data = request.json

        try:
            # add cohort stats by their own and upload from object store
            if data['input_format'] == 'file':
                CohortCharacterstics(inp_form=data['input_format'], input_path=data['input_path'],
                                          observation_ids=data['oids'],selection_criteria=data['selection_criteria'])
            else:
                CohortCharacterstics(inp_form=data['input_format'], db_url=data['db_url'],
                    schema=data['schema'], table=data['table'], observation_ids=data['oids'],selection_criteria=data['selection_criteria'])
        except Exception as e:
            raise exceptions.ParseError("Unable to generate Cohort Statistics for this input")
        return 201
Feature selections algorithm

Recursive Feature Elimination

# Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(dataset.data, dataset.target)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

Feature Importance

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(dataset.data, dataset.target)
# display the relative importance of each attribute
print(model.feature_importances_)


Remove redundant features

# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
data(PimaIndiansDiabetes)
# calculate correlation matrix
correlationMatrix <- cor(PimaIndiansDiabetes[,1:8])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

Rank features by Importance

# ensure results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the dataset
data(PimaIndiansDiabetes)
# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)
# train the model
model <- train(diabetes~., data=PimaIndiansDiabetes, method="lvq", preProcess="scale", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)


Feature Selection

# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# load the data
data(PimaIndiansDiabetes)
# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(PimaIndiansDiabetes[,1:8], PimaIndiansDiabetes[,9], sizes=c(1:8), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
