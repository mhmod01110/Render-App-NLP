import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, dcc
import pandas as pd
import joblib
import plotly.express as px
import re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('stopwords')

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stopwords_arabic = set(stopwords.words('arabic'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_txt)

    def clean_txt(self, text):
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'\b[a-zA-Z]+\b', ' ', text)
        text = re.sub(r'[\U00010000-\U0010ffff]', ' ', text)
        text = re.sub(r':[a-z_]+:', ' ', text)
        text = re.sub('[*?!#@]', ' ', text)
        text = re.sub(r'\|\|+\\s*\d+%\s*\|\|+?[_\-\.\?]+', ' ', text)
        text = re.sub(r'[_\-\.\"\:\;\,\'\،\♡\\\)/(\&\؟]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text_tokens = text.split()
        return ' '.join(text_tokens)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

def custom_joblib_loader(filepath):
    # Import the custom transformer class so it's available during deserialization
    import __main__
    __main__.CustomTransformer = CustomTransformer
    return joblib.load(filepath)

# Load the model
loaded_pipeline = custom_joblib_loader('voting_pipeline_new.pkl')

# Initialize Dash app
app = dash.Dash(external_stylesheets=[dbc.themes.SLATE])
server = app.server

# Define the layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("تخمين اللهجات العربية",  style={'textAlign': 'center', 'marginTop': '50px','marginBottom': '20px', 'color': 'goldenrod', 'fontWeight': 'bold'})
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.H2("(مصري - لبناني - ليبي - مغربي - سوداني)", style={'textAlign': 'center', 'marginBottom': '30px', 'color': 'goldenrod', 'fontWeight': 'bold'})
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Textarea(
                id="text-input",
                placeholder="رجاء إدخال نص عربي",
                style={"width": "100%", "height": "200px", "backgroundColor": "#343a40", "color": "#ffffff"},
                className="dark-textarea"
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Button("تخمين", id="predict-button", color="light", size="lg", className="d-grid gap-2 col-6 mx-auto")
                ], width="auto"),
                dbc.Col([
                    html.Button("مسح", id="reset-button", n_clicks=0, className="btn btn-danger", style={"fontSize": "20px"})
                ], width="auto")
            ], justify="start"),
            html.Div(id="prediction-alert", style={"marginTop": "20px"})
        ], width=6),
        dbc.Col([
            dcc.Graph(id="probability-graph")
        ], width=6)
    ])
])

@app.callback(
    [Output("probability-graph", "figure"),
     Output("prediction-alert", "children"),
     Output("text-input", "value")],
    [Input("predict-button", "n_clicks"),
     Input("reset-button", "n_clicks")],
    [State("text-input", "value")]
)
def update_output(predict_clicks, reset_clicks, text):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "predict-button" and (predict_clicks is None or text is None or text.strip() == ""):
        return dash.no_update, None, text
    elif button_id == "predict-button":
        text_series = pd.Series([text])
        probabilities = loaded_pipeline.predict_proba(text_series)
        target_labels = ['EG', 'LB', 'LY', 'MA', 'SD']
        target_names = ['مصري', 'لبناني', 'ليبي', 'مغربي', 'سوداني']
        label_name_mapping = dict(zip(target_labels, target_names))
        class_labels = loaded_pipeline.classes_
        prob_dict = {label_name_mapping[label]: prob for label, prob in zip(class_labels, probabilities[0])}
        final_prediction_label = loaded_pipeline.predict(text_series)[0]
        final_prediction_name = label_name_mapping[final_prediction_label]
        labels = list(prob_dict.keys())
        probs = list(prob_dict.values())
        fig = px.bar(x=labels, y=probs, labels={'x': 'الدولة', 'y': 'الاحتمال'}, title='نسبة التأكد من التخمين')

        fig.update_traces(marker_color=['goldenrod' if label == final_prediction_name else 'gray' for label in labels],
                          text=probs, texttemplate='%{text:.2f}', textposition='outside', textfont_size=14, textfont_family='Arial', textfont_color='white')
        
        fig.update_layout(height=400, width=600, margin=dict(l=40, r=40, t=40, b=40), 
                          title={'x':0.5, 'xanchor': 'center', 'font': {'size': 20, 'family': 'Arial', 'color': 'white'}},
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          xaxis={'title': {'text': 'الدولة', 'font': {'size': 18, 'family': 'Arial', 'color': 'white'}}, 
                                 'tickfont': {'size': 14, 'family': 'Arial', 'color': 'white'}},
                          yaxis={'title': {'text': 'الاحتمال', 'font': {'size': 18, 'family': 'Arial', 'color': 'white'}}, 
                                 'tickfont': {'size': 14, 'family': 'Arial', 'color': 'white'}})

        alert = dbc.Alert(f"التنبؤ: {final_prediction_name}", color="info", className="d-grid gap-2 col-6 mx-auto", style={'textAlign': 'center', 'marginTop': '50px','marginBottom': '30px', 'color': 'black', 'fontWeight': 'bold', 'fontSize': '20px'})
        return fig, alert, text
    elif button_id == "reset-button":
        return {}, None, ""
    else:
        return dash.no_update, None, text

if __name__ == '__main__':
    app.run_server(debug=True)
