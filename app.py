import os
import torch
from flask import Flask, render_template, Response, request, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from torchvision.transforms import functional as F
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import numpy as np
from datetime import datetime
import glob

app = Flask(__name__)

weights_path = 'yolov5m.pt'
device = select_device('')
model = attempt_load(weights_path, device)
model.eval()

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

# Variable global para guardar el último frame procesado
latest_frame = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_objects(model, device, frame):
    img_size = 640 # Reasignar el tamaño del frame para que sea igual que el tamaño del modelo
    frame = cv2.resize(frame, (img_size, img_size))

    # Preprocesar el fotograma 
    img = F.to_tensor(frame)
    img = img.unsqueeze(0).to(device)

    # Correr el modelo
    with torch.no_grad():
        results = model(img)

    # Post-procesar los resultados
    results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.5)

    # Definir colores para los vehiculos
    class_colors = {
        'car': (0, 255, 0),        # Verde
        'truck': (255, 0, 0),      # Azul
        'bus': (0, 0, 255),        # Rojo
        'motorcycle': (255, 255, 0)  # Amarillo
    }

    # Dibujar cuadros delimitadores en el marco con los colores correspondientes
    for result in results:
        if result is not None and len(result) > 0:
            for obj in result:
                x1, y1, x2, y2, conf, cls = obj
                label = f'{model.names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir las cordenadas a integers

                # Cojer el color
                class_name = model.names[int(cls)]
                color = class_colors.get(class_name, (0, 0, 0))  # Por defecto negro si no se encuentra la clase

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

def generate_frames(video_source):
    if isinstance(video_source, str):
        # Abrir archivo de video
        cap = cv2.VideoCapture(video_source)
    else:
        # Usar la webcam
        cap = cv2.VideoCapture(video_source)

    while True:
        # Leer los frames
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar los objetos en el frame
        frame = detect_objects(model, device, frame)

        # Convertir el frame a una imagen
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for display in HTML
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Comprobar si el archivo se ha subido
        if 'file' not in request.files:
            return render_template('upload.html', error='No file selected')

        file = request.files['file']

        # Comprobar si el archivo es valido
        if file and allowed_file(file.filename):
            # Eliminar los otros videos descargados 
            files_to_delete = glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*'))
            for file_to_delete in files_to_delete:
                os.remove(file_to_delete)
                
            # Generar archivo con la fecha y tiempo
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{current_time}.mp4"

            # Guardar video 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Obtener video descargado
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template('processed_video.html', video_path=video_path)
        else:
            return render_template('upload.html', error='Invalid file format')

    return render_template('upload.html')
    
@app.route('/video_feed_client', methods=['POST'])
def video_feed_client():
    global latest_frame

    # Recivir el frame del cliente
    frame_bytes = request.files['frame'].read()
    frame = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # Procesar el frame
    frame = detect_objects(model, device, frame)

    # Convertir el frame a una imagen
    ret, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    # Actualizar el latest_frame
    latest_frame = frame

    return frame_bytes

@app.route('/processed_frame')
def processed_frame():
    global latest_frame

    if latest_frame is None:
        return Response(status=204)  # Retornar un status 204 'No content' si latest_frame no está disponible

    # Convertir el frame a una imagen
    ret, buffer = cv2.imencode('.jpg', latest_frame)
    frame_bytes = buffer.tobytes()

    return Response(frame_bytes, mimetype='image/jpeg')

@app.route('/client')
def client():
    return render_template('client.html')

@app.route('/')
def index():
    video_path = 'Sam.mp4'  # Especificar la ruta del archivo a renderizar en el index
    return render_template('index.html', video_path=video_path)


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')


@app.route('/video_feed/<path:video_path>')
def video_feed(video_path):
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_webcam',)
def video_feed_webcam():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)

