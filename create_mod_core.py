import pandas as pd
from prophet import Prophet
import pickle
from taipy import Config, Scope
import taipy as tp

def clean_data(initial_dataset: pd.DataFrame):
    print("Cleaning Data")
    initial_dataset = initial_dataset.rename(columns={"Date": "ds", "Close": "y"})
    initial_dataset['ds'] = pd.to_datetime(initial_dataset['ds'])
    cleaned_dataset = initial_dataset.copy()
    return cleaned_dataset

## Input Data Nodes
initial_dataset_cfg = Config.configure_data_node(id="initial_dataset",
                                                 storage_type="csv",
                                                 path='df_aapl.csv',
                                                 scope=Scope.GLOBAL)

cleaned_dataset_cfg = Config.configure_data_node(id="cleaned_dataset",
                                             scope=Scope.GLOBAL)                                                  

clean_data_task_cfg = Config.configure_task(id="clean_data_task",
                                            function=clean_data,
                                            input=initial_dataset_cfg,
                                            output=cleaned_dataset_cfg,
                                            skippable=True)


model_training_cfg = Config.configure_data_node(id="model_output",
                                                storage_type= 'pickle',
                                             scope=Scope.GLOBAL)                                                  

def retrained_model(cleaned_dataset: pd.DataFrame):
    print("Model Retraining")
    model= Prophet()
    model.fit(cleaned_dataset)

    with open('aapl_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

model_training_task_cfg = Config.configure_task(id="model_retraining_task",
                                                  function=retrained_model,
                                                  input=[cleaned_dataset_cfg],
                                                  output=model_training_cfg)



# Create the first pipeline configuration
retraining_model_pipeline_cfg = Config.configure_pipeline(id="model_retraining_pipeline",
                                                  task_configs=[clean_data_task_cfg, model_training_task_cfg])

# # Run of the Taipy Core service
# tp.Core().run()

# # # Create the pipeline
# retrain_pipeline = tp.create_pipeline(retraining_model_pipeline_cfg)
# # Submit the pipeline (Execution)
# tp.submit(retrain_pipeline)