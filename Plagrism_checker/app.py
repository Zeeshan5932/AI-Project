from flask import Flask,render_template,request
import pickle


app = Flask(__name__)



Model = pickle.load(open(r"G:\project\Plagrism_checker\Models\model.pkl" , 'rb'))
tfid = pickle.load(open(r"G:\project\Plagrism_checker\Models\tfidf_vectorizer.pkl", 'rb'))



#create function to detect system
def detect(input_text):
    vectorized_txt = tfid.transform([input_text])
    result = Model.predict(vectorized_txt)
    return "Successfully Plagrism Detected" if result[0] == 1 else "No PLagiarism Detected"


# Start the frame of App

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])

def detect_plagirsm():
    input_text = request.form['text']
    detection_result = detect(input_text)
    return render_template('index.html' , result=detection_result)



#indenetify app

if __name__ == "__main__":
    app.run(debug=True)