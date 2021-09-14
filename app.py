from flask import Flask, render_template, request,jsonify
import pickle
import pandas as pd
import ast
import numpy as np

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def model():
    return render_template('index.html')

@app.route('/batch_model')
def batch_model():
    return render_template('batch.html')


@app.route('/predict',methods=['POST'])
def predict():
    model = pickle.load(open('artifacts/modelv2.pkl', 'rb'))

    if request.method == 'POST':
        try:
            features = [np.array([x for x in request.form.values()])]
            check=0

            col_names = ['state', 'account_length', 'area_code', 'international_plan',
            'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes',
            'total_day_calls', 'total_day_charge', 'total_eve_minutes',
            'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
            'total_night_calls', 'total_night_charge', 'total_intl_minutes',
            'total_intl_calls', 'total_intl_charge',
            'number_customer_service_calls']

            features_df = pd.DataFrame(data=features,columns=col_names)

        except:
            batch_input = request.form['data']

            try:
                features = pd.DataFrame.from_dict(ast.literal_eval(batch_input))
                check=1

                features_df=features[['state', 'account_length', 'area_code', 'international_plan',
                                'voice_mail_plan', 'number_vmail_messages', 'total_day_minutes',
                                'total_day_calls', 'total_day_charge', 'total_eve_minutes',
                                'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
                                'total_night_calls', 'total_night_charge', 'total_intl_minutes',
                                'total_intl_calls', 'total_intl_charge',
                                'number_customer_service_calls']]
            except:
                error_statement = "Wrong input format, Please check it again!"
                return render_template("batch.html", error = error_statement)

        try:
            for x in features_df.columns[~features_df.columns.isin(['state','area_code','international_plan','voice_mail_plan'])]:
                if features_df[x].dtypes!='float64':
                    features_df[x] = features_df[x].astype(float)
        except:
            error_statement="Numerical columns should be of float type but given string!"
            if check==0:
                return render_template("index.html", error = error_statement)
            else:
                return render_template("batch.html", error = error_statement)

        for x in features_df.isna().sum():
            if x!=0:
                error_statement="Missing Values Found!"
                return render_template("batch.html", error = error_statement)

        states = ['OH', 'NJ', 'OK', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT',
                  'NY', 'ID', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'WY', 'HI', 'NH',
                  'AK', 'GA', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'SD',
                  'NC', 'WA', 'MN', 'NM', 'NV', 'DC', 'VT', 'KY', 'ME', 'MS', 'AL',
                  'NE', 'KS', 'TN', 'IL', 'PA', 'CT', 'ND']

        areacode = ['area_code_415', 'area_code_408', 'area_code_510']
        international_plan = ['yes','no']
        voice_mail_plan = ['yes','no']

        temp1=0
        for x in features_df['state'].values:
            for i in states:
                if x==i:
                    temp1=temp1+1
                else:
                    pass
        
        if temp1!=len(features_df):
            error_statement = "Please give a valid State!"
            if check==0:
                return render_template("index.html", error = error_statement)
            else:
                return render_template("batch.html", error = error_statement)
            
        temp2=0
        for x in features_df['area_code'].values:
            for i in areacode:
                if x==i:
                    temp2=temp2+1
                else:
                    pass

        if temp2!=len(features_df):
            error_statement = "Please give a valid Area Code!"
            return render_template("batch.html", error = error_statement)
        
        temp3=0
        for x in features_df['international_plan'].values:
            for i in international_plan:
                if x==i:
                    temp3=temp3+1
                else:
                    pass

        if temp3!=len(features_df):
            error_statement = "international_plan should be either in yes/no"
            return render_template("batch.html", error = error_statement)
        
        temp4=0
        for x in features_df['voice_mail_plan'].values:
            for i in voice_mail_plan:
                if x==i:
                    temp4=temp4+1
                else:
                    pass

        if temp4!=len(features_df):
            error_statement = "voice_mail_plan should be either in yes/no"
            return render_template("batch.html", error = error_statement)
        
        for col in features_df.columns[features_df.dtypes == 'object']:
            features_df[col]=features_df[col].astype('category').cat.codes
        
        predict_prob = model.predict_proba(features_df).tolist()
        threshold_val=0.42
        prediction = (model.predict_proba(features_df)[:,1] >= threshold_val).astype(int).tolist()
        
        if check==0:
            result={
                "predict":prediction[0],
                "predict_prob":round(predict_prob[0][1],2),
                "threshold":threshold_val
                  }
        else:
            arr=[]
            for i in range(0,features_df.shape[0]):
                temp={
                    "predict":prediction[i],
                    "predict_prob":round(predict_prob[i][1],2),
                    "threshold":threshold_val
                     }
                arr.append(temp)

            result = dict(enumerate(np.array(arr), 0))

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
