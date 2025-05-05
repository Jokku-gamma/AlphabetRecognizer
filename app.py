from flask import Flask,request,jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app=Flask(__name__)
model=tf.keras.models.load_model('alphabet_classifier.h5')
alph_map={i:chr(65+i) for i in range(26)}
@app.route('/predict',methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error':'No image provided'}),400
        
        img_file=request.files['image']
        image=Image.open(img_file).convert('L')
        image=image.resize((28,28))
        img_arr=np.array(image)
        img_arr=255-img_arr
        img_arr=img_arr.astype('float32')/255.0
        img_arr=img_arr.reshape(1,28,28,1)
        prediction=model.predict(img_arr)
        pred_cls=np.argmax(prediction,axis=1)[0]
        conf=float(np.max(prediction))

        return jsonify({
            'letter':alph_map[pred_cls],
            'confidence':conf
        })
    except Exception as e:
        return jsonify({'error':str(e)}),500
    
if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)