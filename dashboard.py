from datetime import datetime, date
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import time

# Dashboard functions

def get_df_calories_burned_by_date(df_train_time_exercise_by_date):

   calories_by_date = []

   for i in range(0, len(df_train_time_exercise_by_date)):

      if df_train_time_exercise_by_date['id_exercise'][i] == 'push_up':
         calories_by_date.append(df_train_time_exercise_by_date['Training_time'][i] * 8)

      elif df_train_time_exercise_by_date['id_exercise'][i] == 'bird_dog':
         calories_by_date.append(df_train_time_exercise_by_date['Training_time'][i] * 3.5)

      elif df_train_time_exercise_by_date['id_exercise'][i] == 'forward_lunge':
         calories_by_date.append(df_train_time_exercise_by_date['Training_time'][i] * 3.5)

      elif df_train_time_exercise_by_date['id_exercise'][i] == 'front_plank':
         calories_by_date.append(df_train_time_exercise_by_date['Training_time'][i] * 8)

      elif df_train_time_exercise_by_date['id_exercise'][i] == 'curl_up':
         calories_by_date.append(df_train_time_exercise_by_date['Training_time'][i] * 2.8)

   df_train_time_exercise_by_date = df_train_time_exercise_by_date.copy()
   df_train_time_exercise_by_date['Calories_burned'] = calories_by_date

   return df_train_time_exercise_by_date

def get_df_train_time_exercise_by_date(datasets_list):

   df = pd.DataFrame()
   mins_calculated = []
   for i in datasets_list:
       
      df_temp = pd.read_csv(".//03. users//"+i)
      df_temp2 = df_temp.copy()
      df_temp["DateTime_Start"] = pd.to_datetime(df_temp.DateTime_Start)
      
      first_time = df_temp['DateTime_Start'].iloc[0].time()
      last_time = df_temp['DateTime_Start'].iloc[-1].time()
      time_diff = (datetime.combine(date.today(), last_time) - datetime.combine(date.today(), first_time)).total_seconds()
      mins_calculated.append(round(time_diff / 60,2))
      
      df_temp2['Fecha'] = df_temp2['DateTime_Start'].str.slice(0,10)
      df_temp2 = df_temp2[['id_exercise', 'Fecha']]
      df_temp2 = df_temp2.drop_duplicates()

      df = pd.concat([df, df_temp2])

   df['Training_time'] = mins_calculated
   df = df.groupby(['id_exercise', 'Fecha']).sum()
   df = df.reset_index()
   #del df['index']
   df.columns = ['id_exercise', 'Fecha', 'Training_time']
   #print(df.head())
   return df

def plot_barchar_by_date(df, x, y, color, text, x_axis, y_axis, title):

   st.markdown("<h1 style='text-align: center; font-size: 20px;'>"+title+" </h1>", unsafe_allow_html=True)
   fig = px.bar(df, x=x, y=y, color=color, text=text, barmode='group',)
   fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
   fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
   fig.layout.width = 378
   fig.layout.height = 320
   return fig

def get_df_by_whole_training_data(datasets_list):

   df_whole_training = pd.DataFrame()
   for i in datasets_list:
       
       df_temp = pd.read_csv(".//03. users//"+i)
       df_whole_training = pd.concat([df_whole_training, df_temp])

   return df_whole_training

def get_data_grouped_by_date(df_whole_training):

   df_whole_training['Fecha'] = df_whole_training['DateTime_Start'].str.slice(0,10)

   df_grouoped_by_date = df_whole_training[['id_exercise', 'Fecha']]
   df_grouoped_by_date = df_grouoped_by_date.drop_duplicates()
   df_grouoped_by_date = df_grouoped_by_date.reset_index()
   
   del df_grouoped_by_date['index']

   return df_grouoped_by_date

def get_datestamp_txt():
    return time.strftime("%Y%m%d")

def get_files_by_day(user, day, dir):
   b_today_training = []

   for file in os.listdir(dir):
      if file.startswith(day) and file.endswith(".csv") and user == file[16:23]:
         b_today_training.append(file)
   return b_today_training

def get_files_by_dir(user, dir):
   c_whole_training = []

   for file in os.listdir(dir):
      if file.endswith(".csv") and user == file[16:23]:
         c_whole_training.append(file)
   return c_whole_training

def plot_cal_burned_card_chart(value, title, suffix):
   st.markdown("<h1 style='text-align: center; font-size: 20px;'>Calorías quemadas</h1>", unsafe_allow_html=True)
   fig = go.Figure()

   fig.add_trace(go.Indicator(
      mode = "number",
      value = value,
      number = {'suffix': suffix, 'font': {'size': 20}},
      title = {'text': title, 'font': {'size': 25}},
   ))
   fig.layout.width = 400
   fig.layout.height = 350
   fig.update_layout(margin={'t': 0})
      #fig.show()
   return fig

def plot_training_time_card_chart(value, title, suffix):
   st.markdown("<h1 style='text-align: center; font-size: 20px;'>Tiempo entrenamiento</h1>", unsafe_allow_html=True)
   fig = go.Figure()

   fig.add_trace(go.Indicator(
      mode = "number",
      value = value,
      number = {'suffix': suffix, 'font': {'size': 20}},
      title = {'text': title, 'font': {'size': 25}},
   ))
   fig.layout.width = 400
   fig.layout.height = 350
   fig.update_layout(margin={'t': 0})
      #fig.show()
   return fig

def plot_aprox_gauge_chart(value, title, range_min, range_max, range_step1, range_step2):
      st.markdown("<h1 style='text-align: center; font-size: 20px;'>Aproximación</h1>", unsafe_allow_html=True)
      fig = go.Figure(go.Indicator(
      domain = {'x': [0, 1], 'y': [0, 1]},
      mode = "gauge+number",
      value = value,
      title = {'text': title, 'font': {'size': 21}},
      number = {'suffix': '%', 'font': {'size': 20}},
      gauge = {'axis': {'range': [range_min, range_max], 'tickvals': [range_min, range_max]},
                    'steps' : [
                        {'range': [range_min, range_step1], 'color': "red"},
                        {'range': [range_step1, range_step2], 'color': "yellow"},
                        {'range': [range_step2  , range_max], 'color': "green"}],
                     'bar': {'color': 'black'},
                    'threshold' : {'line': {'color': "green", 'width': 4}, 'thickness': 0.75, 'value': range_max}}))
      fig.layout.width = 400
      fig.layout.height = 350
      fig.update_layout(margin={'t': 0})
      return fig

def get_calories_burned(training_time_exercise, exercise_selected):

   if exercise_selected == 'push_up':

      calories_burned =  training_time_exercise * 8

   elif exercise_selected == 'bird_dog':

      calories_burned =  training_time_exercise * 3.5

   elif exercise_selected == 'forward_lunge':

      calories_burned =  training_time_exercise * 3.5

   elif exercise_selected == 'front_plank':

      calories_burned =  training_time_exercise * 8

   elif exercise_selected == 'curl_up':

      calories_burned =  training_time_exercise * 2.8   

   return calories_burned

def get_training_time(df_data_exercise):

   df_data_exercise["DateTime_Start"] = pd.to_datetime(df_data_exercise.DateTime_Start)
   first_time = df_data_exercise['DateTime_Start'].iloc[0].time()
   last_time = df_data_exercise['DateTime_Start'].iloc[-1].time()
   time_diff = (datetime.combine(date.today(), last_time) - datetime.combine(date.today(), first_time)).total_seconds()
   minutes_diff = round(time_diff / 60,2)

   return minutes_diff;

def get_aprox_exercise(cols_angl_exerc_selected, df_prom_angles, df_data_exercise):

   cols_data_excercise = ['count_pose','id_exercise']

   df_data_exercise = df_data_exercise[cols_data_excercise+cols_angl_exerc_selected]
   df_data_exercise = df_data_exercise.reset_index()
   
   del df_data_exercise['index']

   prom_aprox_angles =[] # eliminar ef de efectividad ya que ahora se denomina aproximacion
   names_angles_prom_matrix = df_prom_angles['Parte'].unique()
   for i in range(0, len(df_data_exercise)):
      print(df_data_exercise['id_exercise'][i]) 
      count_pose = df_data_exercise['count_pose'][i]
      #count_pose=count_pose[0]
      print("Pose: ", count_pose)
      #angles_user = []
      #angles_trainer = []
      #min_angles = []
      #max_angles = []
      aproxs_angles = []

      for j in range(0, len(cols_angl_exerc_selected)):
         print("Pose segun loop: ", str(df_prom_angles['pose'][j]))
         print("Count pose: ", str(count_pose))
         angle_user=df_data_exercise[cols_angl_exerc_selected[j]][i]
         angle_trainer = df_prom_angles['Angulo'][(df_prom_angles['pose']==int(count_pose))&(df_prom_angles['Parte']==names_angles_prom_matrix[j])]
         angle_trainer = angle_trainer.iloc[0]
         desv_angle_trainer = df_prom_angles['Desviacion_estandar_f'][(df_prom_angles['pose']==int(count_pose))&(df_prom_angles['Parte']==names_angles_prom_matrix[j])]
         desv_angle_trainer = desv_angle_trainer.iloc[0]
            
         #angles_trainer.append(angle_trainer)

         min_angle = angle_trainer - desv_angle_trainer
         max_angle = angle_trainer + desv_angle_trainer
      
         if angle_user <= angle_trainer:

            aproxs_angles.append(round(100*abs(((angle_user-min_angle)/(angle_trainer-min_angle))),2))
         else:

            aproxs_angles.append(round(100*(1-((angle_user-angle_trainer)/(max_angle-angle_trainer))),2))

      prom_aprox_angles.append(round(np.mean(aproxs_angles),1))
   return round(np.mean(prom_aprox_angles),1)

def plot_scatter_user_costs(df_recent_trainig):

   st.markdown("<h1 style='text-align: center; font-size: 20px;'>Análisis de tendencia de costos</h1>", unsafe_allow_html=True)
   fig = px.scatter(df_recent_trainig, x="pose_trainer_cost_min", y="pose_trainer_cost_max", color="pose_user_cost")
   #fig.layout.width = 400
   #fig.layout.height = 350

   return fig;