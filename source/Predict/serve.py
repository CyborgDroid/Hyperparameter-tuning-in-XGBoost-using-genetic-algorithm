import mlflow.pyfunc
class XGBWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import xgboost as xgb
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(context.artifacts["xgb_model"])

    def predict(self, context, model_input):
        input_matrix = xgb.DMatrix(model_input.values)
        return self.xgb_model.predict(input_matrix)
        
model = mlflow.sklearn.load_model('mlruns/1/ffd2e55e0d074a4580ec39fd257846be/artifacts/model_1')
