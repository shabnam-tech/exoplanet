from flask import Flask, request, jsonify, render_template, send_file
import sqlite3
import joblib
import pandas as pd
import numpy as np

import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import json
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from datetime import datetime
from db import get_connection, create_table


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return super().default(obj)

app = Flask(__name__)
create_table()

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "habitability_model.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "model_features.pkl"))

# Feature name mapping
feature_names = {
    "pl_rade": "Planet Radius",
    "pl_bmasse": "Planet Mass",
    "pl_orbsmax": "Orbital Distance",
    "st_teff": "Star Temperature",
    "st_met": "Star Metallicity",
    "st_luminosity": "Star Luminosity",
    "pl_luminosity": "Planet Luminosity",
    "stellar_compatibility_index": "Stellar Compatibility Index"
}


def get_db():
    return get_connection()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-ui")
def predict_ui():
    return render_template("predict.html", features=features, feature_names=feature_names)

@app.route("/rank-ui")
def rank_ui():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM planets", conn)
    conn.close()

    if df.empty:
        return render_template("rank.html", data=None)

    X = df[features]
    df["habitability_score"] = model.predict_proba(X)[:, 1]
    ranked = df.sort_values("habitability_score", ascending=False)

    return render_template("rank.html", data=ranked.to_dict(orient="records"))

@app.route("/add_planet", methods=["POST"])
def add_planet():
    data = request.json
    missing = [f for f in features if f not in data]

    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO planets (
            pl_rade, pl_bmasse, pl_orbsmax,
            st_teff, st_met, st_luminosity,
            pl_luminosity, stellar_compatibility_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(data[f] for f in features))

    conn.commit()
    conn.close()

    return jsonify({"message": "Planet added successfully"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    missing = [f for f in features if f not in data]

    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    input_df = pd.DataFrame([data])[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]


    # Save to database
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO planets (
            pl_rade, pl_bmasse, pl_orbsmax,
            st_teff, st_met, st_luminosity,
            pl_luminosity, stellar_compatibility_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, tuple(float(data[f]) for f in features))
    conn.commit()
    conn.close()

    return jsonify({
        "habitability_class": "Habitable" if prediction == 1 else "Non-Habitable",
        "habitability_probability": round(float(probability), 3)
    })

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File must be CSV format"}), 400
    
    try:
        # Read CSV file
        df = pd.read_csv(file)
        
        # Check if all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing columns: {missing_features}"}), 400
        
        # Get only required features
        X = df[features]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Add predictions to dataframe
        df['habitability_class'] = ['Habitable' if p == 1 else 'Non-Habitable' for p in predictions]
        df['habitability_probability'] = np.round(probabilities, 3)
        
        # Save all rows to database
        conn = get_db()
        cursor = conn.cursor()
        for idx, row in df.iterrows():
            cursor.execute("""
                INSERT INTO planets (
                    pl_rade, pl_bmasse, pl_orbsmax,
                    st_teff, st_met, st_luminosity,
                    pl_luminosity, stellar_compatibility_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(float(row[f]) for f in features))
        conn.commit()
        conn.close()
        
        # Create output file
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(output, 
                        mimetype='text/csv',
                        as_attachment=True,
                        download_name='exoplanet_predictions.csv')
    
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 400

@app.route("/rank")
def rank():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM planets", conn)
    conn.close()

    if df.empty:
        return render_template("rank.html", data=None)

    X = df[features]
    scores = model.predict_proba(X)[:, 1]
    df["habitability_score"] = pd.to_numeric(scores, errors="coerce")
    df = df.dropna(subset=["habitability_score"])

    ranked = df.sort_values("habitability_score", ascending=False)

    return render_template("rank.html", data=ranked.to_dict(orient="records"))

@app.route("/export-ui")
def export_ui():
    return render_template("export.html")

@app.route("/export")
def export():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM planets", conn)
    conn.close()

    if df.empty:
        return jsonify({"error": "No data to export"}), 400

    X = df[features]
    df["habitability_score"] = model.predict_proba(X)[:, 1]
    ranked = df.sort_values("habitability_score", ascending=False)
    
    return jsonify({"count": len(ranked), "habitable_count": len(ranked[ranked["habitability_score"] > 0.5])})

@app.route("/export-csv")
def export_csv():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM planets", conn)
    conn.close()

    if df.empty:
        return jsonify({"error": "No data to export"}), 400

    X = df[features]
    df["habitability_score"] = model.predict_proba(X)[:, 1]
    ranked = df.sort_values("habitability_score", ascending=False)
    
    output = BytesIO()
    ranked.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output, 
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name='exoplanet_data.csv')

@app.route("/export-excel")
def export_excel():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM planets", conn)
    conn.close()

    if df.empty:
        return jsonify({"error": "No data to export"}), 400

    X = df[features]
    df["habitability_score"] = model.predict_proba(X)[:, 1]
    ranked = df.sort_values("habitability_score", ascending=False)
    
    output = BytesIO()
    ranked.to_excel(output, index=False, sheet_name='Exoplanets')
    output.seek(0)
    
    return send_file(output, 
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    as_attachment=True,
                    download_name='exoplanet_data.xlsx')

@app.route("/export-pdf")
def export_pdf():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM planets", conn)
    conn.close()

    if df.empty:
        return jsonify({"error": "No data to export"}), 400

    X = df[features]
    df["habitability_score"] = model.predict_proba(X)[:, 1]
    ranked = df.sort_values("habitability_score", ascending=False)
    
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#00d4ff'),
        spaceAfter=12,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#00d4ff'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    title = Paragraph("Exoplanet Habitability Report", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Date and stats
    date_text = Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    total_text = Paragraph(f"<b>Total Planets:</b> {len(ranked)}", styles['Normal'])
    habitable_text = Paragraph(f"<b>Habitable Planets:</b> {len(ranked[ranked['habitability_score'] > 0.5])}", styles['Normal'])
    
    story.append(date_text)
    story.append(total_text)
    story.append(habitable_text)
    story.append(Spacer(1, 0.3*inch))
    
    # Data table
    story.append(Paragraph("Planet Data", heading_style))
    
    # Prepare table data
    table_data = [['Rank', 'Radius', 'Mass', 'Distance', 'Star Temp', 'Metallicity', 'Star Lum', 'Planet Lum', 'Habitability']]
    
    for idx, (_, row) in enumerate(ranked.iterrows(), 1):
        table_data.append([
            str(idx),
            f"{row['pl_rade']:.2f}",
            f"{row['pl_bmasse']:.2f}",
            f"{row['pl_orbsmax']:.2f}",
            f"{row['st_teff']:.0f}",
            f"{row['st_met']:.2f}",
            f"{row['st_luminosity']:.2f}",
            f"{row['pl_luminosity']:.2f}",
            f"{row['habitability_score']:.3f}"
        ])
    
    # Create table
    table = Table(table_data, colWidths=[0.45*inch]*9)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d4ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0a0e27')),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#0a0e27'), colors.HexColor('#1a1f3a')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#6b6eff')),
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    output.seek(0)
    
    return send_file(output, 
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name='exoplanet_data.pdf')

@app.route("/dashboard")
def dashboard():
    try:
        print("DEBUG: Dashboard route called")
        conn = get_db()
        df = pd.read_sql_query("SELECT * FROM planets", conn)
        conn.close()
        
        print(f"DEBUG: Retrieved {len(df)} planets from database")

        if df.empty:
            print("DEBUG: DataFrame is empty")
            return render_template("dashboard.html", 
                                 scatter_json=None, 
                                 box_json=None,
                                 orbit_json=None,
                                 importance_json=None,
                                 stats_json=None)

        # Compute habitability score
        X = df[features]
        print(f"DEBUG: Computing predictions for {len(X)} planets")
        df["habitability_score"] = model.predict_proba(X)[:, 1]

        # Assign habitability class
        df["habitability_class"] = np.where(df["habitability_score"] > 0.5, "Habitable", "Non-Habitable")
        
        print(f"DEBUG: Average habitability: {df['habitability_score'].mean()}")

        # 1. Scatter Plot: Planet Radius vs Mass
        fig_scatter = px.scatter(
            df,
            x="pl_rade",
            y="pl_bmasse",
            color="habitability_score",
            size="pl_luminosity",
            hover_data=["st_teff", "st_met"],
            color_continuous_scale="Viridis",
            title="Planet Radius vs Mass (Colored by Habitability Score)"
        )
        fig_scatter.update_layout(template="plotly_dark", plot_bgcolor="rgba(20,20,30,0.8)", paper_bgcolor="rgba(10,10,20,0.9)")
        scatter_json = json.loads(json.dumps(fig_scatter.to_dict(), cls=NumpyEncoder))
        print("DEBUG: Created scatter chart")

        # 2. Box Plot: Planet Radius by Habitability Class
        fig_box = px.box(
            df,
            x="habitability_class",
            y="pl_rade",
            color="habitability_class",
            title="Planet Radius Distribution by Habitability",
            color_discrete_map={"Habitable":"#00d4ff","Non-Habitable":"#6b6eff"}
        )
        fig_box.update_layout(template="plotly_dark", plot_bgcolor="rgba(20,20,30,0.8)", paper_bgcolor="rgba(10,10,20,0.9)")
        box_json = json.loads(json.dumps(fig_box.to_dict(), cls=NumpyEncoder))
        print("DEBUG: Created box chart")

        # 3. Habitability vs Orbital Distance
        fig_orbit = px.scatter(
            df,
            x="pl_orbsmax",
            y="habitability_score",
            color="habitability_score",
            size="pl_bmasse",
            color_continuous_scale="Viridis",
            title="Habitability Score vs Orbital Distance"
        )
        fig_orbit.update_layout(template="plotly_dark", plot_bgcolor="rgba(20,20,30,0.8)", paper_bgcolor="rgba(10,10,20,0.9)")
        orbit_json = json.loads(json.dumps(fig_orbit.to_dict(), cls=NumpyEncoder))
        print("DEBUG: Created orbit chart")

        # 4. Feature Importance
        importance_json = None
        if hasattr(model.named_steps["classifier"], "feature_importances_"):
            importances = model.named_steps["classifier"].feature_importances_
            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values("Importance", ascending=True)
            
            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(
                y=importance_df["Feature"],
                x=importance_df["Importance"],
                orientation="h",
                marker=dict(
                    color=importance_df["Importance"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=importance_df["Importance"].round(4),
                textposition="outside"
            ))
            fig_importance.update_layout(
                title="<b>Feature Importance in Habitability Prediction</b>",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                template="plotly_dark",
                plot_bgcolor="rgba(20, 20, 30, 0.8)",
                paper_bgcolor="rgba(10, 10, 20, 0.9)",
                font=dict(color="white", size=11),
                hovermode="y",
                height=400
            )
            importance_json = json.loads(json.dumps(fig_importance.to_dict(), cls=NumpyEncoder))
            print("DEBUG: Created importance chart")

        # 5. Key statistics
        stats = {
            "total_planets": int(len(df)),
            "avg_habitability": round(float(df["habitability_score"].mean()), 3),
            "max_habitability": round(float(df["habitability_score"].max()), 3),
            "habitable_count": int(len(df[df["habitability_score"] > 0.5]))
        }
        stats_json = stats
        print(f"DEBUG: Stats computed: {stats}")

        return render_template("dashboard.html",
                               scatter_json=scatter_json,
                               box_json=box_json,
                               orbit_json=orbit_json,
                               importance_json=importance_json,
                               stats_json=stats_json)
    except Exception as e:
        print(f"ERROR in dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template("dashboard.html", 
                             scatter_json=None, 
                             box_json=None,
                             orbit_json=None,
                             importance_json=None,
                             stats_json=None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

