import gradio as gr
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

# Load the saved full pipeline from the file
# Load the saved full pipeline from the file
full_pipeline_path = '/workspaces/heart-disease-22-classification/logistic_regression_model.pkl'

with open(full_pipeline_path, 'rb') as f_in:
    full_pipeline = pickle.load(f_in)

# Define the predict function
# Define the predict function
def predict(Sex, GeneralHealth, PhysicalHealthDays, MentalHealthDays,
       LastCheckupTime, PhysicalActivities, SleepHours, RemovedTeeth,
       HadHeartAttack, HadAngina, HadStroke, HadAsthma,
       HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease,
       HadArthritis, HadDiabetes, DeafOrHardOfHearing,
       BlindOrVisionDifficulty, DifficultyConcentrating,
       DifficultyWalking, DifficultyDressingBathing, DifficultyErrands,
       SmokerStatus, ECigaretteUsage, ChestScan, RaceEthnicityCategory,
       AgeCategory, HeightInMeters, WeightInKilograms, BMI,
       AlcoholDrinkers, HIVTesting, FluVaxLast12, PneumoVaxEver,
       TetanusLast10Tdap, HighRiskLastYear, CovidPos):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Sex' : [Sex] if Sex else [0],
        'GeneralHealth' : [GeneralHealth] if GeneralHealth else [0],
        'PhysicalHealthDays' : [PhysicalHealthDays] if PhysicalHealthDays else [0],
        'MentalHealthDays': [Sex] if Sex else [0],
        'LastCheckupTime' : [Sex] if Sex else [0],
        'PhysicalActivities' : [Sex] if Sex else [0],
        'SleepHours' :[Sex] if Sex else [0],
        'RemovedTeeth': [Sex] if Sex else [0],
        'HadHeartAttack': [Sex] if Sex else [0],
        'HadAngina' : [Sex] if Sex else [0],
    
        'HadStroke' : [Sex] if Sex else [0],
        'HadAsthma': [Sex] if Sex else [0],
        'HadSkinCancer' : [Sex] if Sex else [0],
        'HadCOPD' : [Sex] if Sex else [0],
        'HadDepressiveDisorder' : [Sex] if Sex else [0],
        'HadKidneyDisease': [Sex] if Sex else [0],
        'HadArthritis' : [Sex] if Sex else [0],
        'HadDiabetes' : [Sex] if Sex else [0],
        'DeafOrHardOfHearing': [Sex] if Sex else [0],
    
        'BlindOrVisionDifficulty' : [Sex] if Sex else [0],
        'DifficultyConcentrating': [Sex] if Sex else [0],
        'DifficultyWalking' : [Sex] if Sex else [0],
        'DifficultyDressingBathing': [Sex] if Sex else [0],
        'DifficultyErrands': [Sex] if Sex else [0],
        'SmokerStatus': [Sex] if Sex else [0],
        'ECigaretteUsage': [Sex] if Sex else [0],
    
        'ChestScan': [Sex] if Sex else [0],
        'RaceEthnicityCategory': [Sex] if Sex else [0],
        'AgeCategory': [Sex] if Sex else [0],
        'HeightInMeters': [Sex] if Sex else [0],
        'WeightInKilograms' : [Sex] if Sex else [0],
        'BMI': [Sex] if Sex else [0],
        'AlcoholDrinkers': [Sex] if Sex else [0],
        'HIVTesting' : [Sex] if Sex else [0],
    

        'FluVaxLast12': [Sex] if Sex else [0],
        'PneumoVaxEver': [Sex] if Sex else [0],
        'TetanusLast10Tdap': [Sex] if Sex else [0],
        'HighRiskLastYear': [Sex] if Sex else [0],
        'CovidPos': [Sex] if Sex else [0],
    })

        # Make predictions using the loaded logistic regression model
        #predict probabilities
    predictions = full_pipeline.predict_proba(input_data)
    #take the index of the maximum probability
    index=np.argmax(predictions)
    higher_pred_prob=round((predictions[0][index])*100)


    #return predictions[0]
    print(f'[Info] Predicted probabilities{predictions},{full_pipeline.classes_}')
    
    return f'{full_pipeline.classes_[index]} with {higher_pred_prob}% confidence'
    
# Setting Gradio App Interface
with gr.Blocks(css=".gradio-container {background-color:grey }",theme=gr.themes.Base(primary_hue='blue'),title='Uriel') as demo:
    gr.Markdown("# Heart Disease Prediction #\n*This App allows the user to predict whether the patient has heart disease by entering values in the given fields. Any field left blank takes the default value.*")
    gr.Markdown("Sex: 0 for Female, 1 for Male")
    gr.Markdown('GeneralHealth: 0 for Excellent, 1 for Fair , 2 for Good , 3 for Poor, 4 for Very good')
    gr.Markdown('Last Checkup Time: 0 for 5 or more years ago, 1 for Within past 2 years (1 year but less than 2 years ago), 2 for Within past 5 years (2 years but less than 5 years ago), 3 for Within past year (anytime less than 12 months ago)  ')
    gr.Markdown('PhysicalActivities: 1 for Yes, 0 for No')
    gr.Markdown('RemovedTeeth: 0 for 1 to 5,  1 for 6 or more, but not all  , 2 for All , 3 for None of them ')
    gr.Markdown('HadAngina: 0 for no, 1 for Yes')
    gr.Markdown('HadStroke: 0 for no, 1 for yes')
    gr.Markdown('HadAsthma, HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease, HadArthritis, DeafOrHardOfHearing, BlindOrVisionDifficulty, DifficultyConcentrating, DifficultyWalking, DifficultyDressingBathing, DifficultyErrands, ChestScan, HIVTesting, PneumoVaxEver,HighRiskLastYear , AlcoholDrinkers, FluVaxLast12: 0 for no, 1 for yes')
    gr.Markdown('HadDiabetes: 0 for no, 1 for No, pre-diabetes or borderline diabetes  , 2 for Yes  , 3 for Yes, but only during pregnancy (female) ')
    gr.Markdown('SmokerStatus: 0 for Current smoker - now smokes every day, 1 for Current smoker - now smokes some days, 2 for Former smoker   , 3 for Never smoked ')
    gr.Markdown('ECigaretteUsage: 0 for Never used e-cigarettes in my entire life, 1 for Not at all (right now)  , 2 for Use them every day   , 3 for Use them some days   ')
    gr.Markdown('RaceEthnicityCategory: 0 for Black only, Non-Hispanic, 1 for Hispanic , 2 for Multiracial, Non-Hispanic , 3 for Other race only, Non-Hispanic, 4 for White only, Non-Hispanic')
    gr.Markdown('AgeCategory: Age 65 to 69     :9  \
                    Age 60 to 64    :8   \
                    Age 70 to 74    :10   \
                    Age 55 to 59      :7 \
                    Age 50 to 54       :6\
                    Age 75 to 79       :11\
                    Age 80 or older    :12\
                    Age 40 to 44       :4\
                    Age 45 to 49       :5\
                    Age 35 to 39       :3\
                    Age 30 to 34       :2\
                    Age 18 to 24       :0\
                    Age 25 to 29       :1')
    gr.Markdown('TetanusLast10Tdap: No, did not receive any tetanus shot in the past 10 years    :0\
                    Yes, received tetanus shot but not sure what type            :3\
                    Yes, received Tdap                                          :1\
                    Yes, received tetanus shot, but not Tdap :2')
    gr.Markdown('CovidPos:\
                    No                                                               :0\
                    Yes                                                              :2\
                    Tested positive using home test without a health professional:1')
    # Receiving ALL Input Data here
    gr.Markdown("**Demographic Data**")
    with gr.Row():
        Sex = gr.Number()
        GeneralHealth = gr.Number()
        PhysicalHealthDays = gr.Number()
        MentalHealthDays= gr.Number()
        LastCheckupTime = gr.Number()
        PhysicalActivities = gr.Number()
        SleepHours = gr.Number()
        RemovedTeeth= gr.Number()
        HadHeartAttack= gr.Number()
        HadAngina = gr.Number()
    with gr.Row():
        HadStroke = gr.Number()
        HadAsthma= gr.Number()
        HadSkinCancer = gr.Number()
        HadCOPD = gr.Number()
        HadDepressiveDisorder = gr.Number()
        HadKidneyDisease= gr.Number()
        HadArthritis = gr.Number()
        HadDiabetes = gr.Number()
        DeafOrHardOfHearing= gr.Number()
    with gr.Row():
        BlindOrVisionDifficulty = gr.Number()
        DifficultyConcentrating= gr.Number()
        DifficultyWalking = gr.Number()
        DifficultyDressingBathing= gr.Number()
        DifficultyErrands= gr.Number()
        SmokerStatus= gr.Number()
        ECigaretteUsage= gr.Number()
    with gr.Row():
        ChestScan= gr.Number()
        RaceEthnicityCategory= gr.Number()
        AgeCategory= gr.Number()
        HeightInMeters= gr.Number()
        WeightInKilograms = gr.Number()
        BMI= gr.Number()
        AlcoholDrinkers= gr.Number()
        HIVTesting = gr.Number()
    with gr.Row():

        FluVaxLast12= gr.Number()
        PneumoVaxEver= gr.Number()
        TetanusLast10Tdap= gr.Number()
        HighRiskLastYear= gr.Number()
        CovidPos= gr.Number()

    # Output Prediction
    output = gr.Text(label="Outcome")
    submit_button = gr.Button("Predict")
    
    submit_button.click(fn= predict,
                        outputs= output,
                        inputs=[Sex, GeneralHealth, PhysicalHealthDays, MentalHealthDays,
       LastCheckupTime, PhysicalActivities, SleepHours, RemovedTeeth,
       HadHeartAttack, HadAngina, HadStroke, HadAsthma,
       HadSkinCancer, HadCOPD, HadDepressiveDisorder, HadKidneyDisease,
       HadArthritis, HadDiabetes, DeafOrHardOfHearing,
       BlindOrVisionDifficulty, DifficultyConcentrating,
       DifficultyWalking, DifficultyDressingBathing, DifficultyErrands,
       SmokerStatus, ECigaretteUsage, ChestScan, RaceEthnicityCategory,
       AgeCategory, HeightInMeters, WeightInKilograms, BMI,
       AlcoholDrinkers, HIVTesting, FluVaxLast12, PneumoVaxEver,
       TetanusLast10Tdap, HighRiskLastYear, CovidPos]
    
    ),
    
    # Add the reset and flag buttons
    
    def clear():
        output.value = ""
        return 'Predicted values have been reset'
         
    clear_btn = gr.Button("Reset", variant="primary")
    clear_btn.click(fn=clear, inputs=None, outputs=output)
        
 
demo.launch(inbrowser = True)