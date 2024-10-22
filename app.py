from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, Response
from keras.models import load_model
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import io 
import numpy as np
import pydicom
from skimage.filters import gaussian
from skimage import exposure
from skimage.transform import resize
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import credentials, firestore
from contornos import contornos_anotados 
from datetime import datetime
import matplotlib

# Usar el backend Agg para Matplotlib
matplotlib.use('Agg')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'your_secret_key'

# Configurar Firebase
cred = credentials.Certificate('deteccion-85097-firebase-adminsdk-ewhnm-fc59134dc8.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Cargar el modelo
model = load_model('modelo_nodulos_pulmonares.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def preprocesar_imagen_dicom(file_path):
    try:
        dicom_data = pydicom.dcmread(file_path)
        if not hasattr(dicom_data, 'pixel_array'):
            raise ValueError('Archivo no contiene datos DICOM válidos')
        image = dicom_data.pixel_array

        # Verificar que la imagen sea en escala de grises (1 canal)
        if len(image.shape) != 2:
            raise ValueError('El archivo DICOM no tiene un formato de imagen válido')
        
        # Aplicar filtro gaussiano para suavizar la imagen
        image_smooth = gaussian(image, sigma=1)
        # Ajustar el contraste de la imagen
        image_rescale = exposure.rescale_intensity(image_smooth, in_range='image', out_range=(0, 1))
        # Redimensionar la imagen a 128x128 y asegurar que tenga 1 canal
        image_rescale = resize(image_rescale, (128, 128), anti_aliasing=True)
        image_rescale = image_rescale[..., np.newaxis]  # Agregar un canal

        return image_rescale, image.shape, dicom_data.pixel_array  # También devolver la imagen original para visualización

    except Exception as e:
        raise ValueError('El archivo proporcionado no es un archivo DICOM válido')
  # También devolver la imagen original para visualización=


@app.route('/')
def home():
    if 'user' in session:
        # Obtener todos los pacientes registrados
        pacientes_ref = db.collection('pacientes')
        pacientes_docs = pacientes_ref.stream()
        pacientes = [doc.to_dict() for doc in pacientes_docs]
        return render_template('index.html', pacientes=pacientes)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        user_ref = db.collection('users').document(email)
        user_ref.set({
            'email': email,
            'password': hashed_password
        })
        
        flash('Registro exitoso. Ahora puedes iniciar sesión.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user_ref = db.collection('users').document(email)
        user = user_ref.get()
        if user.exists:
            user_data = user.to_dict()
            if check_password_hash(user_data['password'], password):
                session['user'] = email
                flash('Inicio de sesión exitoso.', 'success')
                return redirect(url_for('home'))
            else:
                flash('Contraseña incorrecta.', 'danger')
        else:
            flash('El usuario no existe.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Has cerrado sesión.', 'success')
    return redirect(url_for('login'))

@app.route('/upload_multiple_historial', methods=['GET', 'POST'])
def upload_multiple_files_historial():
    if request.method == 'POST':
        files = request.files.getlist('files')
        paciente_id = request.form.get('paciente_id')  # Obtener el paciente_id del formulario

        if not files or not paciente_id:
            return redirect(request.url)

        resultados = []  # Lista para almacenar los resultados de todas las imágenes

        for file in files:
            if file.filename == '':
                continue

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                img_preprocesada, original_shape, original_image = preprocesar_imagen_dicom(filepath)
                img_preprocesada = np.expand_dims(img_preprocesada, axis=0)
                prediccion = model.predict(img_preprocesada)[0][0]

                resultado = 'Nódulo' if prediccion > 0.5 else 'No Nódulo'
                confianza = prediccion if prediccion > 0.5 else 1 - prediccion

                # Dibujar contornos en la imagen original solo si se detecta un nódulo
                plt.imshow(original_image, cmap='gray')

                if resultado == 'Nódulo':
                    for paciente in contornos_anotados:
                        if filename in contornos_anotados[paciente]:
                            for contour_points in contornos_anotados[paciente][filename]:
                                scaled_contour_points = contour_points * [original_image.shape[1] / original_shape[1], original_image.shape[0] / original_shape[0]]
                                plt.plot(scaled_contour_points[:, 0], scaled_contour_points[:, 1], 'r', linewidth=2)  # Dibujar cada contorno en rojo

                plt.title(f'Predicción: {resultado} (Confianza: {confianza:.2f}) - {filename}')
                plt.axis('off')

                # Guardar la imagen con anotaciones
                annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + filename + '.png')
                plt.savefig(annotated_image_path)
                plt.close()

                # Guardar en Firestore
                doc_ref = db.collection('predicciones').document()
                doc_ref.set({
                    'paciente_id': paciente_id,
                    'filename': filename,
                    'resultado': resultado,
                    'confianza': float(confianza),
                    'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Agregar la fecha y hora
                    'user': session['user'],  # Agregar referencia al usuario
                    'annotated_image': 'annotated_' + filename + '.png'  # Guardar la ruta de la imagen anotada
                })

                resultados = sorted(resultados, key=lambda r: r['confianza'], reverse=True)

                # Agregar resultado a la lista de resultados
                resultados.append({
                    'filename': filename,
                    'resultado': resultado,
                    'confianza': confianza,
                    'annotated_image': 'annotated_' + filename + '.png',
                    'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Agregar la fecha y hora
                })

                

            except ValueError as e:
                flash(str(e), 'danger')
                return redirect(url_for('home'))

        return render_template('resultados_multiples.html', resultados=resultados, paciente_id=paciente_id)
    else:
        pacientes_ref = db.collection('pacientes')
        pacientes_docs = pacientes_ref.stream()
        pacientes = [doc.to_dict() for doc in pacientes_docs]
        return render_template('index.html', pacientes=pacientes)

@app.route('/diagnostico', methods=['GET', 'POST'])
def diagnostico():

    # Obtener todos los pacientes registrados
    pacientes_ref = db.collection('pacientes')
    pacientes_docs = pacientes_ref.stream()
    pacientes = [doc.to_dict() for doc in pacientes_docs]

    return render_template('diagnostico.html', pacientes=pacientes)




@app.route('/historial')
def historial():
    if 'user' in session:
        # Obtener todos los pacientes registrados
        pacientes_ref = db.collection('pacientes')
        pacientes_docs = pacientes_ref.stream()
        pacientes = [doc.to_dict() for doc in pacientes_docs]
        
        # Obtener todas las predicciones
        predicciones_ref = db.collection('predicciones')
        predicciones_docs = predicciones_ref.stream()
        predicciones = [doc.to_dict() for doc in predicciones_docs]

        # Organizar las predicciones por paciente
        historial_pacientes = {}
        for paciente in pacientes:
            paciente_id = paciente['nombre'] + ' ' + paciente['apellido']
            historial_pacientes[paciente_id] = {
                'info': paciente,
                'comorbilidades': paciente.get('comorbilidades', []),  # Asegurarse de obtener las comorbilidades
                'predicciones': [pred for pred in predicciones if pred['paciente_id'] == paciente_id]
            }

        return render_template('historial.html', historial_pacientes=historial_pacientes)
    return redirect(url_for('login'))


@app.route('/registro_paciente', methods=['GET', 'POST'])
def registro_paciente():
    if 'user' in session:
        if request.method == 'POST':
            nombre = request.form['nombre']
            apellido = request.form['apellido']
            fecha_nacimiento = request.form['fecha_nacimiento']
            genero = request.form['genero']
            comorbilidades = request.form.getlist('comorbilidades')  # Obtener lista de comorbilidades seleccionadas
            otra_comorbilidad = request.form.get('otra_comorbilidad')  # Obtener comorbilidad personalizada si la hay
            
            # Si se proporciona otra comorbilidad, añadirla a la lista
            if otra_comorbilidad:
                comorbilidades.append(otra_comorbilidad)
            
            paciente_id = nombre + ' ' + apellido
            
            paciente_ref = db.collection('pacientes').document(paciente_id)
            paciente_ref.set({
                'nombre': nombre,
                'apellido': apellido,
                'fecha_nacimiento': fecha_nacimiento,
                'genero': genero,
                'comorbilidades': comorbilidades  # Guardar comorbilidades
            })
            
            flash('Paciente registrado exitosamente.', 'success')
            return redirect(url_for('home'))
        
        return render_template('registro_paciente.html')
    return redirect(url_for('login'))



@app.route('/editar_paciente/<paciente_id>', methods=['GET', 'POST'])
def editar_paciente(paciente_id):
    if 'user' in session:
        paciente_ref = db.collection('pacientes').document(paciente_id)
        paciente = paciente_ref.get().to_dict()

        if request.method == 'POST':
            nombre = request.form['nombre']
            apellido = request.form['apellido']
            fecha_nacimiento = request.form['fecha_nacimiento']
            genero = request.form['genero']

            paciente_ref.update({
                'nombre': nombre,
                'apellido': apellido,
                'fecha_nacimiento': fecha_nacimiento,
                'genero': genero
            })

            flash('Paciente actualizado exitosamente.', 'success')
            return redirect(url_for('historial'))

        return render_template('editar_paciente.html', paciente=paciente)
    return redirect(url_for('login'))

@app.route('/listado_pacientes')
def listado_pacientes():
    if 'user' in session:
        search_query = request.args.get('search', '').lower()  # Obtener el valor de búsqueda
        pacientes_ref = db.collection('pacientes')
        pacientes_docs = pacientes_ref.stream()
        pacientes = [doc.to_dict() for doc in pacientes_docs]
        
        # Filtrar pacientes si hay búsqueda
        if search_query:
            pacientes = [paciente for paciente in pacientes if search_query in (paciente['nombre'] + ' ' + paciente['apellido']).lower()]
        
        return render_template('listado_pacientes.html', pacientes=pacientes)
    return redirect(url_for('login'))



@app.route('/eliminar_paciente/<paciente_id>', methods=['POST'])
def eliminar_paciente(paciente_id):
    if 'user' in session:
        paciente_ref = db.collection('pacientes').document(paciente_id)
        paciente_ref.delete()

        # También eliminamos todas las predicciones asociadas a este paciente
        predicciones_ref = db.collection('predicciones').where('paciente_id', '==', paciente_id)
        for prediccion in predicciones_ref.stream():
            prediccion.reference.delete()

        flash('Paciente eliminado exitosamente.', 'success')
        return redirect(url_for('historial'))
    return redirect(url_for('login'))

@app.route('/reporte_paciente/<paciente_id>')
def reporte_paciente(paciente_id):
    paciente_ref = db.collection('pacientes').document(paciente_id)
    paciente = paciente_ref.get().to_dict()

    predicciones_ref = db.collection('predicciones').where('paciente_id', '==', paciente_id)
    predicciones_docs = predicciones_ref.stream()
    predicciones = [doc.to_dict() for doc in predicciones_docs]

    # Obtener las comorbilidades del paciente, si existen
    comorbilidades = paciente.get('comorbilidades', [])

    return render_template('reporte_paciente.html', paciente=paciente, predicciones=predicciones, comorbilidades=comorbilidades)


@app.route('/reportes')
def reportes():
    if 'user' in session:
        # Obtener todos los pacientes registrados
        pacientes_ref = db.collection('pacientes')
        pacientes_docs = pacientes_ref.stream()
        pacientes = [doc.to_dict() for doc in pacientes_docs]
        
        # Obtener todas las predicciones
        predicciones_ref = db.collection('predicciones')
        predicciones_docs = predicciones_ref.stream()
        predicciones = [doc.to_dict() for doc in predicciones_docs]
        
        # Filtrar pacientes con nódulos
        pacientes_con_nodulos = {}
        for paciente in pacientes:
            paciente_id = paciente['nombre'] + ' ' + paciente['apellido']
            predicciones_paciente = [pred for pred in predicciones if pred['paciente_id'] == paciente_id and pred['resultado'] == 'Nódulo']
            if predicciones_paciente:
                # Obtener las comorbilidades del paciente si existen
                comorbilidades = paciente.get('comorbilidades', [])
                pacientes_con_nodulos[paciente_id] = {
                    'info': paciente,
                    'predicciones': predicciones_paciente,
                    'comorbilidades': comorbilidades
                }
        
        return render_template('reportes.html', pacientes_con_nodulos=pacientes_con_nodulos)
    return redirect(url_for('login'))
    

@app.route('/grafico_pacientes_nodulos')
def grafico_pacientes_nodulos():
    # Obtener todas las predicciones
    predicciones_ref = db.collection('predicciones')
    predicciones_docs = predicciones_ref.stream()
    predicciones = [doc.to_dict() for doc in predicciones_docs]
    
    # Contar pacientes con y sin nódulos
    con_nodulo = sum(1 for pred in predicciones if pred['resultado'] == 'Nódulo')
    sin_nodulo = sum(1 for pred in predicciones if pred['resultado'] == 'No Nódulo')
    
    # Generar gráfico
    labels = ['Con Nódulo', 'Sin Nódulo']
    sizes = [con_nodulo, sin_nodulo]
    colors = ['#ff9999','#66b3ff']
    explode = (0.1, 0)  # solo "explode" el primer slice
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Igualar el eje X e Y para que el gráfico sea un círculo perfecto
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return Response(img, mimetype='image/png')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)





