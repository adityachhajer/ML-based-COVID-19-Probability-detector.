from flask import Flask,render_template,request
app = Flask(__name__)
import pickle

file = open('model.pkl', 'rb')
clf=pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        myDict=request.form
        fever=int(myDict['fever'])
        pain=int(myDict['pain'])
        age=int(myDict['age'])
        runnyNose=int(myDict['diffBreath'])
        diffBreath=int(myDict['diffBreath'])

        inputFeatures = [fever, pain, age, runnyNose, diffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][0]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))


    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)