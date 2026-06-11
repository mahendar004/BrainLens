from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import shutil
import cv2

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload folder
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load CNN model
MODEL_PATH = "cnn_stroke.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "cnn_stroke.h5 not found in project folder."
    )

model = load_model(MODEL_PATH)


# ------------------------------------
# PDF REPORT GENERATOR
# ------------------------------------

def generate_report(normal, stroke, guidance):

    pdf_file = "BrainLens_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "BrainLens Stroke Detection Report",
            styles['Title']
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            f"<b>Normal Probability:</b> {normal}",
            styles['Normal']
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            f"<b>Stroke Probability:</b> {stroke}",
            styles['Normal']
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "<b>Recommended Action:</b>",
            styles['Heading2']
        )
    )

    elements.append(
        Paragraph(
            guidance.replace("\n", "<br/>"),
            styles['Normal']
        )
    )

    doc.build(elements)

    return pdf_file


# ------------------------------------
# GUIDANCE FUNCTION
# ------------------------------------

def get_guidance(normal_prob, stroke_prob):

    if stroke_prob > 0.7:

        return (
            "High probability of stroke detected.\n"
            "Call emergency services immediately.\n"
            "Do not wait.\n"
            "Keep the patient calm and airway clear."
        ), "emergency"

    elif stroke_prob > 0.4:

        return (
            "Moderate probability of stroke.\n"
            "Seek medical evaluation as soon as possible.\n"
            "Monitor symptoms closely."
        ), "moderate"

    else:

        return (
            "Low probability of stroke.\n"
            "Maintain a healthy lifestyle:\n"
            "Balanced diet, exercise and regular checkups."
        ), "normal"


# ------------------------------------
# CT SCAN VALIDATION
# ------------------------------------

def is_ct_scan(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return False

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    saturation_mean = np.mean(hsv[:, :, 1])

    if saturation_mean > 70:
        return False

    mean_intensity = np.mean(gray)

    std_intensity = np.std(gray)

    if (
        mean_intensity < 20
        or mean_intensity > 230
        or std_intensity < 15
    ):
        return False

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    edge_density = (
        np.sum(edges > 0)
        / (gray.shape[0] * gray.shape[1])
    )

    if edge_density < 0.01 or edge_density > 0.3:
        return False

    return True


# ------------------------------------
# HOME PAGE
# ------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    filename = None

    if request.method == "POST":

        if "file" not in request.files:
            return "No file selected"

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        # Delete previous uploads
        for f in os.listdir(app.config['UPLOAD_FOLDER']):

            path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                f
            )

            try:

                if os.path.isfile(path):
                    os.unlink(path)

            except Exception as e:
                print(e)

        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            file.filename
        )

        file.save(filepath)

        filename = file.filename

        # Validate CT Scan
        if not is_ct_scan(filepath):

            result = {
                "guidance":
                "Invalid image. Upload a valid Brain CT scan."
            }

            return render_template(
                "index.html",
                result=result,
                filename=None
            )

        try:

            img = Image.open(filepath)

            img = img.convert("RGB")

            img = img.resize((224, 224))

            img_array = img_to_array(img)

            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            prediction = model.predict(img_array)

            stroke_prob = float(prediction[0][0])

            normal_prob = 1 - stroke_prob

            guidance_text, guidance_class = get_guidance(
                normal_prob,
                stroke_prob
            )

            result = {

                "normal":
                f"{normal_prob * 100:.2f}%",

                "stroke":
                f"{stroke_prob * 100:.2f}%",

                "guidance":
                guidance_text,

                "guidance_class":
                guidance_class
            }

            # Store values for PDF
            app.config["NORMAL_RESULT"] = result["normal"]
            app.config["STROKE_RESULT"] = result["stroke"]
            app.config["GUIDANCE_RESULT"] = result["guidance"]

        except Exception as e:

            print(e)

            result = {
                "guidance":
                "Error processing image."
            }

    return render_template(
        "index.html",
        result=result,
        filename=filename
    )


# ------------------------------------
# DOWNLOAD REPORT
# ------------------------------------

@app.route("/download_report")
def download_report():

    pdf_file = generate_report(

        app.config.get(
            "NORMAL_RESULT",
            "N/A"
        ),

        app.config.get(
            "STROKE_RESULT",
            "N/A"
        ),

        app.config.get(
            "GUIDANCE_RESULT",
            "No Guidance"
        )
    )

    return send_file(
        pdf_file,
        as_attachment=True
    )


# ------------------------------------
# RUN APP
# ------------------------------------

if __name__ == "__main__":
    app.run(debug=True)