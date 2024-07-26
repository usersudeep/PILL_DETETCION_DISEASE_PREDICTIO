import json
from flask import Flask, redirect, render_template, request, session, url_for
from chat_bot_n import computeresult, listdesises, otherSymptoms
import os
import unittest
from verify import detect_pill_from_img
from io import BytesIO 
#app = Flask(__name__, template_folder='static', static_folder='static')
app = Flask(__name__,template_folder="C:/Users/ssude/Downloads/pill_helth_proj-20231112T064823Z-001/pill_helth_proj/static") 

@app.route('/')
def hello_world():
	return render_template('index.html')

@app.route('/disease')
def disease():
	return render_template('decese_ind.html')

@app.route('/pill')
def pill():
	return render_template('pillo_ind.html')

@app.route('/detected_pill', methods = ['POST'])   
def detected_pill():
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save(f.filename)          
        print("Saved the file",f.filename)
        pill_name, per = detect_pill_from_img(f.filename)
        fname_name_percent= []
        fname_name_percent.append(f.filename)
        fname_name_percent.append(pill_name)
        fname_name_percent.append(per)
        return render_template("detected_pill.html",fname_name_percent = fname_name_percent)  

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form['fname']
        semptom = request.form['smptm']
        deseses = listdesises(semptom,user)
        if deseses == 0:
             return render_template('index.html')
        else:
            return render_template('desese.html' , deseses = deseses)
    else:
        user = request.form['fname']
        semptom = request.form['smptm']
        return  render_template('desese.html' )
    
@app.route('/symptom' , methods=['GET', 'POST'])
def symptom():
    if request.method == 'POST':
        user_answer=request.form['op1']   
        ndays1 = request.form['ds1']
        ndays = int (ndays1)
        ans1 = user_answer
        ans2 = ans1.removesuffix('"')
        ans = ans2.removeprefix(' "')
        print("sympt:")
        print(ans)
        print("No:Days:")
        print(ndays)
        otrsympt = otherSymptoms(ans,ndays)
        print(otrsympt)
        return render_template('symptom.html',otrsympt = otrsympt)
    else:
        return  render_template('symptom.html' )

@app.route('/result' , methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        symptoms_exp1 = request.form.getlist("cb1")
        itms = []
        for itm in symptoms_exp1:
            tmp =  itm.replace('"','')
            tmp1= tmp.removeprefix(" ").removesuffix(" ")
            itms.append(tmp1)
        rslt = computeresult(itms)
        return render_template('result.html',rslt = rslt )
    else:
        return  render_template('result.html' )
    
if __name__ == '__main__':

	# run() method of Flask class runs the application 
	# on the local development server.
	app.run(debug=True)
