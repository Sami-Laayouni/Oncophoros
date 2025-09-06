# Quick user interface

import os
import json
import joblib
import pandas as pd
import gradio as gr
import numpy as np
from sklearn.preprocessing import StandardScaler


MODEL_CONFIGS = {
    "LightGBM": {
        "dir": "results_lightgbm",
        "model_file": "calibrated_model.joblib",
        "needs_scaling": False,
        "scaler_file": None
    },
    "Logistic Regression": {
        "dir": "results_lr", 
        "model_file": "calibrated_model.joblib",
        "needs_scaling": True,
        "scaler_file": "scaler.joblib"
    },
    "Random Forest": {
        "dir": "results_rf",
        "model_file": "calibrated_rf_model.joblib",
        "needs_scaling": False,
        "scaler_file": None
    }
}


class MultiModelPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.all_features = []
        self.feature_medians = {}
        self.model_summaries = {}
        self.load_all_models()
    
    def load_all_models(self):
        loaded_models = []
        for model_name, config in MODEL_CONFIGS.items():
            try:
                model_dir = config["dir"]
                model_path = os.path.join(model_dir, config["model_file"])
                if not os.path.exists(model_path):
                    continue
                self.models[model_name] = joblib.load(model_path)
                if config["needs_scaling"] and config["scaler_file"]:
                    scaler_path = os.path.join(model_dir, config["scaler_file"])
                    if os.path.exists(scaler_path):
                        self.scalers[model_name] = joblib.load(scaler_path)
                    else:
                        del self.models[model_name]
                        continue
                summary_path = os.path.join(model_dir, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        self.model_summaries[model_name] = json.load(f)
                loaded_models.append(model_name)
            except:
                if model_name in self.models:
                    del self.models[model_name]
        if not self.models:
            raise Exception("No models could be loaded")
        self.load_features(loaded_models[0])
    
    def load_features(self, model_name):
        config = MODEL_CONFIGS[model_name]
        model_dir = config["dir"]
        processed_file = os.path.join(model_dir, "processed_features_and_labels.parquet")
        if os.path.exists(processed_file):
            df_proc = pd.read_parquet(processed_file)
            feature_cols = [c for c in df_proc.columns if c.startswith('feat_')]
            self.all_features = [c.replace('feat_', '') for c in feature_cols]
            for col in feature_cols:
                name = col.replace('feat_', '')
                self.feature_medians[name] = df_proc[col].median()
        else:
            self.all_features = [f"feature_{i}" for i in range(100)]
            self.feature_medians = {f: 0.0 for f in self.all_features}
    
    def get_top_features(self, model_name, n_features=15):
        config = MODEL_CONFIGS[model_name]
        feature_path = os.path.join(config["dir"], "top20_features.csv")
        if os.path.exists(feature_path):
            df = pd.read_csv(feature_path)
            return df['feature'].tolist()[:n_features]
        else:
            return self.all_features[:n_features]

    def predict_single_model(self, model_name, feature_inputs):
        try:
            if model_name not in self.models:
                return None, "Model not available"
            input_data = {}
            for fname, val in feature_inputs.items():
                if fname in self.all_features:
                    input_data[fname] = float(val) if val != "" else self.feature_medians.get(fname, 0.0)
            for feature in self.all_features:
                if feature not in input_data:
                    input_data[feature] = self.feature_medians.get(feature, 0.0)
            df_input = pd.DataFrame([input_data])
            df_input = df_input.reindex(columns=self.all_features, fill_value=0.0)
            if model_name in self.scalers:
                df_input = pd.DataFrame(
                    self.scalers[model_name].transform(df_input),
                    columns=df_input.columns
                )
            model = self.models[model_name]
            risk_prob = model.predict_proba(df_input)[0, 1]
            risk_pred = model.predict(df_input)[0]
            return risk_prob, "HBV" if risk_pred == 1 else "Non-Viral"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def predict_all_models(self, **feature_inputs):
        results = {}
        for model_name in self.models.keys():
            prob, pred = self.predict_single_model(model_name, feature_inputs)
            results[model_name] = {
                "probability": prob,
                "prediction": pred,
                "risk_level": self.get_risk_level(prob) if prob is not None else "Error"
            }
        return results

    def get_risk_level(self, prob):
        if prob < 0.3:
            return "Low Risk"
        elif prob < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"

    def get_available_models(self):
        return list(self.models.keys())


try:
    predictor = MultiModelPredictor()
    models_loaded = True
    available_models = predictor.get_available_models()
except Exception as e:
    print(f"Failed to load models: {e}")
    models_loaded = False
    available_models = []


def create_interface():
    if not models_loaded:
        with gr.Blocks(title="HBV Predictor - Error") as demo:
            gr.Markdown("# Model Loading Error")
            gr.Markdown("Could not load any models")
        return demo

    with gr.Blocks(title="Multi-Model HBV Risk Predictor") as demo:
        gr.Markdown("Comparison of Different Models Accuracy")
        
        with gr.Row():
            with gr.Column(scale=2):
                model_choice = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="Select Modle for Input"
                )
                feature_inputs = {}
                initial_features = predictor.get_top_features(available_models[0]) if available_models else []
                for feature in initial_features:
                    default_val = predictor.feature_medians.get(feature, 0.0)
                    feature_inputs[feature] = gr.Number(
                        label=f"{feature}",
                        placeholder=f"Default: {default_val:.3f}",
                        value=None
                    )
                predict_btn = gr.Button("Predict with All Models")
                if predictor.model_summaries:
                    perf_data = []
                    for model_name in available_models:
                        if model_name in predictor.model_summaries:
                            metrics = predictor.model_summaries[model_name].get('metrics', {})
                            perf_data.append([
                                model_name,
                                f"{metrics.get('ROC-AUC', 'N/A'):.3f}" if isinstance(metrics.get('ROC-AUC'), (int, float)) else 'N/A',
                                f"{metrics.get('F1', 'N/A'):.3f}" if isinstance(metrics.get('F1'), (int, float)) else 'N/A',
                                f"{metrics.get('Accuracy', 'N/A'):.3f}" if isinstance(metrics.get('Accuracy'), (int, float)) else 'N/A'
                            ])
                    if perf_data:
                        gr.Dataframe(
                            value=perf_data,
                            headers=["Model", "ROC-AUC", "F1-Score", "Accuracy"],
                            interactive=False
                        )
            with gr.Column(scale=1):
                model_results = {}
                for model_name in available_models:
                    gr.Markdown(f"### {model_name}")
                    model_results[model_name] = {
                        "text": gr.Markdown("*Click Predict to see results*"),
                        "prob": gr.Slider(
                            minimum=0, maximum=1, value=0,
                            label="Risk Probability",
                            interactive=False
                        )
                    }
                gr.Markdown("This tool is only for the Hackathon TurBioHacks (NO MEDICAL USAGE)")
        
        def make_predictions(*args):
            current_features = predictor.get_top_features(available_models[0])
            feature_dict = {}
            for i, feature in enumerate(current_features):
                if i < len(args) and args[i] is not None:
                    feature_dict[feature] = args[i]
                else:
                    feature_dict[feature] = ""
            results = predictor.predict_all_models(**feature_dict)
            outputs = []
            for model_name in available_models:
                if model_name in results:
                    result = results[model_name]
                    prob = result["probability"]
                    pred = result["prediction"]
                    if prob is not None:
                        result_text = f"{pred} | {result['risk_level']} | {prob:.1%}"
                        outputs.extend([result_text, float(prob)])
                    else:
                        outputs.extend([f"Error: {pred}", 0.0])
                else:
                    outputs.extend(["Not available", 0.0])
            return outputs

        all_outputs = []
        for model_name in available_models:
            all_outputs.extend([
                model_results[model_name]["text"],
                model_results[model_name]["prob"]
            ])
        predict_btn.click(
            fn=make_predictions,
            inputs=list(feature_inputs.values()),
            outputs=all_outputs
        )

        def update_features(selected_model):
            if not selected_model or selected_model not in available_models:
                return [gr.update() for _ in feature_inputs.values()]
            new_features = predictor.get_top_features(selected_model)
            updates = []
            for i, (old_feature, component) in enumerate(feature_inputs.items()):
                if i < len(new_features):
                    new_feature = new_features[i]
                    default_val = predictor.feature_medians.get(new_feature, 0.0)
                    updates.append(gr.update(
                        label=new_feature,
                        placeholder=f"Default: {default_val:.3f}"
                    ))
                else:
                    updates.append(gr.update())
            return updates

        model_choice.change(
            fn=update_features,
            inputs=[model_choice],
            outputs=list(feature_inputs.values())
        )

    return demo

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False
    )
