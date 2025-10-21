import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data, preprocess_data
from evaluation import evaluate_model


def load_config(config_path='config.yaml'):
    """Carga configuración desde YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_pipeline():
    """Pipeline completo de entrenamiento"""
    # Cargar configuración
    config = load_config()

    # Configurar MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Iniciar run de MLflow
    with mlflow.start_run():
        print("Iniciando pipeline de ML...")

        # 1. Cargar datos
        print("Cargando datos...")
        df = load_data(config['data']['url'])

        # 2. Preprocesamiento
        print("Preprocesando datos...")
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
            df,
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state']
        )

        # 3. Entrenar modelo
        print("Entrenando modelo...")
        model = RandomForestClassifier(**config['model']['params'])
        model.fit(X_train, y_train)

        # 4. Evaluar modelo
        print("Evaluando modelo...")
        metrics = evaluate_model(model, X_test, y_test)

        # 5. Registrar en MLflow
        print("Registrando en MLflow...")

        # Parámetros
        mlflow.log_params(config['model']['params'])
        mlflow.log_param("test_size", config['data']['test_size'])

        # Métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")

        # Signature y ejemplo de entrada
        from mlflow.models.signature import infer_signature

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_test[:5]

        # Guardar modelo
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example,
            registered_model_name="wine-quality-classifier"
        )

        print("Pipeline completado exitosamente!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_pipeline()
