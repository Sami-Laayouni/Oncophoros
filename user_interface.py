import gradio as gr
import joblib
import pandas as pd
import numpy as np

model = joblib.load(r"C:\Users\Admin\Desktop\IGEM Hackathon\results_lightgbm\calibrated_model.joblib")
top_feats = pd.read_csv(r"C:\Users\Admin\Desktop\IGEM Hackathon\results_lightgbm\top20_features.csv")["feature"].tolist()

def predict(inputs):
    df = pd.DataFrame([inputs], columns=top_feats)
    df = np.log2(df+1)  # same preprocessing
    prob = model.predict_proba(df)[:,1][0]
    return {"HBV probability": float(prob)}

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=f) for f in top_feats],
    outputs="label"
)
demo.launch()
