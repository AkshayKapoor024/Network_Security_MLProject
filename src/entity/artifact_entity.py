from dataclasses import dataclass
# Output of DataIngestion Process stored in this class
@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

# Output of DataValidation Process stored in this class
@dataclass
class DataValidationArtifact:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str
# Output of Data Transformation Process stored in this class
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    transformed_object_file_path:str
    
@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
        
# Output of Model Trainer Process stored in this class
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    train_metric_artifact:ClassificationMetricArtifact
    test_metric_artifact:ClassificationMetricArtifact
    