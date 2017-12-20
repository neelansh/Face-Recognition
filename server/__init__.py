import os
from flask import Flask, render_template, request

from face_recog import *

app = Flask(__name__, template_folder='templates', static_url_path='/static', static_folder='static')

UPLOAD_FOLDER = os.path.basename('uploads')
TEMPLATE_FOLDER = os.path.basename('templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATE_FOLDER'] = TEMPLATE_FOLDER


@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():
	if(request.method == 'GET'):
		return render_template('index.html')

	elif(request.method == 'POST'):
	    file = request.files['image']
	    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
	    
	    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
	    file.save(f)

	    num_faces, faces = recog_face_openface(f, os.path.join(os.getcwd(), 'static','output', file.filename), request.form['classifier'])

	    return render_template('index.html', num_faces = num_faces, faces= faces, file_name=file.filename);

if __name__ == "__main__":
    app.run()
