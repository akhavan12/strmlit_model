################################################################################################
### import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sts
pd.options.display.max_rows = 999
import matplotlib.cm as cm
import streamlit as st
################################################################################################
import sys

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



#import SessionState

#session = SessionState.get(code='Show Flowchart')
#a = st.radio("Show/Hide Flowchart", ['Hide', 'Show'], 0)
#if a == 'Show':
#    st.markdown("![alt text](https://raw.githubusercontent.com/akhavan12/discrete_covid_model/master/Model_Discrete_v4_new.svg)")
#else:
#    st.markdown(body=' ')













def get_country_data(country='Italy'):
    """ input country name
      output df_confirmed,df_recovered,df_dead
    """
    df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    df_dead = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    df_confirmed = df_confirmed[(df_confirmed['Country/Region'].str.lower()==country.lower()) & (df_confirmed['Province/State'].isnull())].T  
    df_confirmed.columns=['confirmed']
    df_recovered = df_recovered[(df_recovered['Country/Region'].str.lower()==country.lower()) & (df_recovered['Province/State'].isnull())].T  
    df_recovered.columns=['recovered']
    df_dead = df_dead[(df_dead['Country/Region'].str.lower()==country.lower()) & (df_dead['Province/State'].isnull())].T  
    df_dead.columns = ['dead']
    return {'df_confirmed':df_confirmed.iloc[4:],'df_recovered':df_recovered.iloc[4:],'df_dead':df_dead.iloc[4:]}


################################################################################################
def get_actual_ts(key='confirmed',country='Italy',rolling=4,rolling_sum=10,population=60_000_000):
  """
    retrive informaton from actual cases database
    Parameters: 
    key (str): "confirmed", "recovered" or "dead"
    country (str): defult is italy (not case sensetive)
    rolling (int): rolling average for the daily cases
  
    Returns: 
    pandas dataframe: based on the type of the key
  
  """
  ### based on the key the data file is loaded 
  if key == "confirmed":
    df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
  elif key == "recovered":
    df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
  elif key == "dead":
    df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
  else:
    return -1

  ###

  df_=df[(df['Country/Region'].str.lower()==country.lower()) & (df['Province/State'].isnull())].T  
  df_ = df_.iloc[4:]
  df_ = df_.set_index(pd.to_datetime(df_.index))

  df_.columns=['cumulative'] 

  df_['daily'] = df_['cumulative'] - df_['cumulative'].shift(1).fillna(0)
  
  df_['rolling'] = df_['daily'].rolling(rolling).mean().fillna(0)
  df_['daily'] = (df_['daily']/population) 

  df_['cumulative_percent']= (df_['cumulative'] / population)
 
  df_['active']=df_['daily'].rolling(rolling_sum).sum()

  return df_

# Our World in Data
data_OWD = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
OWD = pd.read_csv(data_OWD)

list_of_countries = list(OWD[(OWD['new_tests'].notnull()) & OWD['location'] != 'Italy']['location'].unique())
list_of_countries.insert(0,'Italy')


############################### Controls   ##########################################
#####################################################################################
#####################################################################################
selected_country = st.sidebar.selectbox(
    'Countries:',
        list_of_countries)
#st.write( OWD[OWD['new_tests'].notnull()]['location'].unique())

map_type = st.sidebar.selectbox(
    'Figure types:',
    ('Subplots', 'Seperate Plots')
)
SHIFT =                      st.sidebar.slider('Shift Actual Data - to adjust start date', 2, 100,   46)
B1=                          st.sidebar.slider('β 1 (initial transmission rate)', 0.05, 0.90, 0.50)
B2_start=                    st.sidebar.slider('β change point (intervention start date)', 2, 100,   32)
B2=                          st.sidebar.slider('β 2 (after intervention transmission rate)', .05, 0.90,0.13)
m_to_r =                     st.sidebar.slider('Mild to Recovery rate', 0.01, 0.80, 0.49) # m_to_r
w_to_r_ratio=                st.sidebar.slider('Severe to Recovery rate', 0.01,1.0, 0.66) # w_to_r_ratio
w_to_v_ratio=                st.sidebar.slider('Severe to Ventilator rate', 0.01,1.0,0.65) #w_to_v_ratio 
v_to_r_ratio=                st.sidebar.slider('Ventilator to Recovery rate',0.01,1.0,0.0900 ) # v_to_r_ratio
m_v50 =                      st.sidebar.slider('Mild to Recovery Sigmoid V50 point (day)', 1, 14,7)
w_v50 =                      st.sidebar.slider('Severe to Recovery Sigmoid V50 point (day)' , 1, 14, 5),
v_v50 =                      st.sidebar.slider('Ventilator to Recovery Sigmoid V50 point (day)' , 1, 14, 6),
testing_ratio_symp_1 =         st.sidebar.slider('Initial Testing Ratio', 0.01, 0.40, 0.03,)
testing_ratio_symp_2_start= st.sidebar.slider('Point of testing regime change (day)', 2, 130 ,45)
testing_ratio_symp_2 =         st.sidebar.slider('Secondary Testing Ratio', 0.01, 0.99, 0.35)
Beta_other =                 st.sidebar.slider('β other (NONE COVID-19) transmission rate)', 0.001, 0.09,  0.01)
Y_Axis = st.sidebar.selectbox(
    'Y_Axis',
    ('per Population', 'Count')
)



############################### Population ##########################################
#####################################################################################
#####################################################################################
population = OWD[OWD['location'] == selected_country ]

df_recovered = get_country_data(country=selected_country)['df_recovered']

##############
##############

prc = 2 ## Precision for the calculations
T = []

confirmed_ = []
confirmed_daily = []

def set_probs_Other_Exposed(Probs):
  ### Other_Exp,Other_Symp
  Probs["Other_Exposed"] = [
          [1.00,0.00], #0
          [0.80,0.20], #1
          [0.60,0.40], #2
          [0.30,0.70], #3
          [0.00,1.00], #4
  ]
def set_probs_Other_Symp(Probs):
  ### Other_Symp,Susceptible   
  Probs['Other_Symp'] =[[0.80,  0.20],
                        [0.50,  0.50 ], 
                        [0.00,  1.00]]      

def set_probs_M(Probs,num_days = 14, alpha_r=.8, v50=5):
  ### M,R,Severe
  def m_at_day(num_days = 14, alpha_r = .8, v50=5):
      ## https://www.graphpad.com/guides/prism/7/curve-fitting/reg_classic_boltzmann.htm
      Tdays = np.arange(num_days) #x
      Top = 1
      Buttom = 0
      slope= 1.8
      m_to_m = Top +  (Buttom - Top)/(1+ np.exp((v50 - Tdays)/slope))
      m_to_r = alpha_r*(1 - m_to_m)
      m_to_w = 1 - (m_to_m+m_to_r)
      return pd.DataFrame({'m_to_m': m_to_m, 'm_to_r' :m_to_r,'m_to_w': m_to_v})
  
  Probs["M"] = m_at_day(num_days = num_days , alpha_r = alpha_r, v50=v50).values

def set_probs_MU(Probs,num_days = 14, alpha_r=.8, v50=5):
  ### M,R,Severe
  def m_at_day(num_days = 14, alpha_r = .8, v50=5):
      ## https://www.graphpad.com/guides/prism/7/curve-fitting/reg_classic_boltzmann.htm
      Tdays = np.arange(num_days) #x
      Top = 1
      Buttom = 0
      slope= 1.8
      m_to_m = Top +  (Buttom - Top)/(1+ np.exp((v50 - Tdays)/slope))
      m_to_r = alpha_r*(1 - m_to_m)
      m_to_w = 1 - (m_to_m+m_to_r)
      return pd.DataFrame({'m_to_m': m_to_m, 'm_to_r' :m_to_r,'m_to_w': m_to_w})
  
  Probs["M_undiagnosed"] = m_at_day(num_days = num_days , alpha_r = alpha_r, v50=v50).values

def set_probs_Severe(Probs,num_days=14,alpha_r = .4,alpha_v = .2, v50= 7):
  ### Severe,R,V,D

  def w_at_day(num_days = 14,alpha_r = .4,alpha_v = .2):
      ## https://www.graphpad.com/guides/prism/7/curve-fitting/reg_classic_boltzmann.htm
      Tdays = np.arange(num_days) #x
      Top = 1
      Buttom = 0
      v50= num_days/3
      slope= 1.8
      w_to_w = Top +  (Buttom - Top)/(1+ np.exp((v50 - Tdays)/slope))

      w_to_r = alpha_r*(1 - w_to_w)
      w_to_v = alpha_v*(1 - w_to_w)
      w_to_d = 1 - (w_to_w+w_to_r+w_to_v)
      return pd.DataFrame({'w_to_w': w_to_w, 
                           'w_to_r' :w_to_r, 
                           'w_to_v' :w_to_v,
                           'w_to_d': w_to_d})
  Probs["Severe"] = w_at_day(num_days = num_days,alpha_r = alpha_r,alpha_v = alpha_v).values

def set_probs_ventilator(Probs, num_days=20, v_to_r_ratio=.4, v50=10):
  ### V,R,D 
  def v_at_day(num_days = 20,alpha_r = .4, v50=10):
      ## https://www.graphpad.com/guides/prism/7/curve-fitting/reg_classic_boltzmann.htm
      Tdays = np.arange(num_days) #x
      Top = 1
      Buttom = 0
      slope= 1.8
      v_to_v = Top +  (Buttom - Top)/(1+ np.exp((v50 - Tdays)/slope))
      v_to_r = alpha_r*(1 - v_to_v)
      v_to_d = 1 - (v_to_v+v_to_r)
      return pd.DataFrame({'v_to_v': v_to_v, 'v_to_r' :v_to_r,'v_to_d': v_to_d})
  Probs["ventilator"] = v_at_day(num_days = num_days ,alpha_r =v_to_r_ratio,v50=v50).values

def set_probs_Symp(Probs):
  ### Symp,M   0: not tested, 1: tested positive ,2: tested negative
  Probs['Symp'] =[[0.20,  0.80 , 0.00 ], #0 Testing and + cases go to Mild cases , negative cases go back to Healthy
                  [0.00,  0.60 , 0.40 ]]      #1 ###  all remainings go to Mild cases 

def set_probs_E(Probs):
  ### E,Symp
  Probs["E"] = [[1.00,0.00], #0 ### first day of exposure
                [0.50,0.50], #1       
                [0.40,0.60], #2 
                [0.30,0.70], #4
                [0.20,0.80], #5
                [0.20,0.80], #6
                [0.20,0.80], #7
                [0.20,0.80], #8
                [0.20,0.80], #9
                [0.20,0.80], #10
                [0.05,0.95], #11
                [0.05,0.95], #12
                [0.05,0.95], #13
                [0.05,0.95], #14
                [0.05,0.95], #15
                [0.05,0.95], #16
                [0.05,0.95], #17
                [0.05,0.95], #18
                [0.05,0.95], #19
                [0.00,1.00]  #20
                ]

  def e_at_day(num_days = 20):
      ## https://www.graphpad.com/guides/prism/7/curve-fitting/reg_classic_boltzmann.htm
      Tdays = np.arange(num_days) #x
      Top = 1
      Buttom = 0
      v50= 7
      slope= 1.8
      e_to_e = Top +  (Buttom - Top)/(1+ np.exp((v50 - Tdays)/slope))
      e_to_m  = 1 - (e_to_e)
      return pd.DataFrame({'e_to_e': e_to_e, 'e_to_m' :e_to_m})
  Probs["E"] = e_at_day(20).values


############
###########
def get_FR_with_testing(OWD=OWD,country='Italy'):
  ITL_OWD = OWD[OWD['location']==country].copy()
  #ITL_OWD=ITL_OWD[['date','total_cases','new_cases','total_deaths','new_deaths','new_tests','total_tests']].copy()
  ITL_OWD=ITL_OWD[['date','population','new_cases','total_cases','new_deaths','total_deaths','new_tests','new_tests_per_thousand']].copy()
  ITL_OWD['date']=pd.to_datetime(ITL_OWD['date'])
  ITL_OWD.fillna(0,inplace=True)
  ITL_OWD.set_index('date',drop=True,inplace=True)
  Others = ITL_OWD['new_tests'] - ITL_OWD['new_cases']
  ITL_OWD['Death_Ratio']=(ITL_OWD['total_deaths']/ITL_OWD['total_cases']).fillna(0)
  return ITL_OWD






#############
##############
###
P={}
EXP_Other =[]
MU = []     ## Mild undiagnosed
E = []       ## Exposed 
H = []      ## Healthy (susceptible)
Symp = []   ## Symptomatic
SYMP_Other = []
M = []      ## Mild Symptomatic
W = []      ## Severe
V = []       ## Ventilator
R = []      ## Recovered
D = []      ## Dead

Testing = []
add_M = []  
add_sym = [] 
add_E = [] 
add_Severe = []
add_ventilator = []

Track_M = []
Track_E = []
Track_V = []
Track_W = []

other_dis_ratio = .01


def calculate(initial_population = 1000,
              initial_exposed = 1,
              lenght_t = 100,
              B1=0.1,
              B2_start=30,
              B2=.1,
              B = 0.1,
              m_days=15,
              w_days=15,
              v_days=15,
              v_to_r_ratio = 0.3,
              w_to_r_ratio = 0.1,
              w_to_v_ratio = 0.1,
              m_to_r = 0.8,
              m_v50=7,
              w_v50=7,
              v_v50=7,
              testing_ratio_symp_1=0.030,
              testing_ratio_symp_2_start=50,
              testing_ratio_symp_2=0.030,
              Beta_other= 0.04              
              ): 

  
  global P
  global EXP_Other
  global SYMP_Other
  global confirmed_
  global confirmed_daily
  global MU     ## Mild undiagnosed
  global E      ## Exposed 
  global H      ## Healthy (susceptible)
  global Symp   ## Symptomatic
  global M      ## Mild Symptomatic
  global W      ## Severe
  global V      ## Ventilator
  global R      ## Recovered
  global D      ## Dead
  global add_M  
  global add_sym 
  global add_E 
  global add_Severe
  global add_ventilator
  global Track_M
  global Track_E
  global Track_V
  global Track_W
  global Testing


  lenght_t  = int(lenght_t)
  Beta =np.concatenate([np.repeat(B1,B2_start),np.repeat(B2,lenght_t-B2_start)])

  #### Concatinate the testing ratios in different times ######
  testing_ratio_symp = np.concatenate(
      [np.repeat(testing_ratio_symp_1,int(testing_ratio_symp_2_start)),
      np.repeat(testing_ratio_symp_2,lenght_t-int(testing_ratio_symp_2_start))])



  Track_M = []
  Track_E = []
  Track_V = []
  Track_W = []

  P={}
  set_probs_Severe(P,num_days=w_days,alpha_r=w_to_r_ratio,alpha_v=w_to_v_ratio,v50=w_v50)
  set_probs_ventilator(P,num_days=v_days,v_to_r_ratio=v_to_r_ratio,v50=v_v50)
  set_probs_M(P,num_days=m_days,alpha_r=m_to_r,v50=m_v50)
  ## a bit more fixed than the rest of the variables

  set_probs_Symp(P)
  set_probs_E(P)

  #### Other deiseases variables: NOT COVID
  set_probs_Other_Exposed(P)
  set_probs_Other_Symp(P)
  EXP_Other = np.zeros((len(P['Other_Exposed']),lenght_t))
  SYMP_Other = np.zeros((len(P['Other_Symp']),lenght_t))
  
  #### COVID variables
  set_probs_MU(P) ### undiagnosed MILD COVID
  MU = np.zeros((14,lenght_t))
  E = np.zeros((len(P['E']),lenght_t))
  Symp = np.zeros((len(P['Symp']),lenght_t))
  M = np.zeros((len(P['M']),lenght_t))
  W = np.zeros((len(P['Severe']),lenght_t))
  V = np.zeros((len(P['ventilator']),lenght_t))
  R = np.zeros(lenght_t)
  D = np.zeros(lenght_t)
  H = np.zeros(lenght_t)

  Testing = np.zeros(lenght_t)

  confirmed_ = np.zeros(lenght_t)
  confirmed_daily = np.zeros(lenght_t)

  add_M = []
  add_sym = []
  add_E = []
  add_Severe = []
  add_ventilator = []
############################################################
############################################################
############################################################
  #print(E.shape)
  E[0,0] = initial_exposed
  M[0,0] = 0
############################################################
############################################################
############################################################
  H[0] = 1000   #initial_population# - E[0,0]

  for t in range(lenght_t-1):

    ##### Other exposed parameters  
    ##### Other_Symp

    for state in range(len(P['Other_Exposed'])):
      try:
        EXP_Other[state+1,t+1] =  np.round(EXP_Other[state,t] * P["Other_Exposed"][state][0] ,prc) ## Stay exposed conditions
        Testing[t] = Testing[t] + EXP_Other[state+1,t] * testing_ratio_symp[t]

      except:
        pass
      SYMP_Other[0,t+1] = np.round(SYMP_Other[0,t+1] + EXP_Other[state,t] * P["Other_Exposed"][state][1],prc)  ## at time t+1 add the portion of the Exposed to the symptomatic
    

    R[t+1] = R[t]
    D[t+1] = D[t]

    for state in range(len(P['E'])):
      try:
        E[state+1,t+1] =  np.round(E[state,t] * P["E"][state][0],prc) ## Stay exposed conditions
      except:
        pass
      Symp[0,t+1] = np.round(Symp[0,t+1] + E[state,t] * P["E"][state][1],prc)  ## at time t+1 add the portion of the Exposed to the symptomatic
    
    
    ### Symptomatic states Tests take place here
    #################################################                            0      1 
    Symp[1,t+1] = np.round(Symp[0,t] * P["Symp"][0][0],prc)         ### P["Symp"][0][0] ==> [[0,1]  [0,1]]
                                                      ###                      |   |   |  |   
                                                      ###         next day    Symp M
                                                      ###         next day             Symp M
    Testing[t] = Testing[t] + Symp[0,t] * testing_ratio_symp[t]
    ### from Symptomatic to mild --> where testing also happens
    M[1,t+1] = np.round(Symp[0,t] * P["Symp"][0][1],prc)                 ### From state 0 to Mild state 0
    M[1,t+1] = np.round(M[1,t+1] +  Symp[1,t] * P["Symp"][1][1],prc)     ### From state 1 to Mild state 0

    Track_M.append(M[:,t])

    MU[0,t+1] = np.round(Symp[0,t] * P["Symp"][0][2],prc)                 ### From state 0 to Mild state 0
    MU[0,t+1] = np.round(MU[0,t+1] +  Symp[1,t] * P["Symp"][1][2],prc)     ### From state 1 to Mild state 0
    Testing[t] = Testing[t] + MU[0,t]


    #################################################
    #add_sym.append(Symp[0,t] - (Symp[0,t] * P["Symp"][0][0]+Symp[0,t] * P["Symp"][0][1]))
    #add_sym.append(Symp[1,t] -  Symp[1,t] * P["Symp"][1][1] )

    ### Mild cases are calculated in this section
    for state in range(1,len(P['M'])):
      ### M,R,Severe
      try:
        M[state+1,t+1] =  np.round(M[state,t] * P["M"][state][0],prc) ## from state= state --> to state+1
      except:
        pass
      R[t+1] = np.round(R[t+1] + M[state,t] * P["M"][state][1],prc) ## from Mild to recovered
      W[0,t+1] = W[0,t+1] + np.round(M[state,t] * P["M"][state][2],prc)
      Testing[t] = Testing[t] + W[0,t+1]

    ###############################################
    ### Un-diagnosed Mild 
    for state in range(0,len(P['M_undiagnosed'])):
      ### M,R,Severe
      try:
        MU[state+1,t+1] =  np.round(MU[state,t] * P["M_undiagnosed"][state][0],prc) ## from state= state --> to state+1
      except:
        pass

      R[t+1] = np.round(R[t+1] + MU[state,t] * P["M_undiagnosed"][state][1],prc) ## from undiagnosed_Mild to recovered
      #if MU[state,t] * P["M_undiagnosed"][state][1]>0:
        #print(state, t,':', MU[state,t] * P["M_undiagnosed"][state][1])

    ##########################################
    ### Severe Cases 

    for state in range(0,len(P['Severe'])):
    ### Severe,R,V,D
      try:
        W[state+1,t+1] =  np.round(W[state,t] * P["Severe"][state][0],prc)
      except:
        pass
      R[t+1] = np.round(R[t+1] + W[state,t] * P["Severe"][state][1],prc)

      if W[state,t] * P["Severe"][state][2]>0:
        V[0,t+1] = V[0,t+1] + np.round(W[state,t] * P["Severe"][state][2],prc)

      D[t+1] = D[t+1] + np.round(W[state,t] * P["Severe"][state][3],prc)

    ###########################################
    ### ventilator cases

    for state in range(0,len(P['ventilator'])):
    ### V,R,D 
      try: 
        V[state+1,t+1] = np.round( V[state,t] * P["ventilator"][state][0],prc)
      except:
        pass
      R[t+1] = R[t+1] + np.round(V[state,t] * P["ventilator"][state][1],prc)
      D[t+1] = D[t+1] + np.round(V[state,t] * P["ventilator"][state][2],prc)


    EXP_Other[0,t+1] = H[t]* Beta_other
    M=np.array(M)
    if H[t] >(H[t]/initial_population)  *  M[:,t].sum() * Beta[t]:
      E[0,t+1] = (H[t]/initial_population)  *  M[:,t].sum() * Beta[t]
      H[t+1]= H[t] - np.round(E[0,t+1])
      confirmed_[t+1]=confirmed_[t] + (H[t]/initial_population)  *  M[:,t].sum() * Beta[t]
      confirmed_daily[t] = (H[t]/initial_population)  *  M[:,t].sum() * Beta[t]
    elif ((H[t]>0) & ((H[t]/initial_population)  *  M[:,t].sum() * Beta[t])):
      E[0,t+1] = H[t]
      H[t+1]= 0


    Track_E.append(E[:,t])
    Track_V.append(V[:,t])
    Track_W.append(W[:,t])

  return D

############
############

# Add a selectbox to the sidebar:
#location = st.sidebar.selectbox(
#    'Select the country to analyze?',
#    'Italy'
#)







calculate(initial_population=1000,
          initial_exposed=1,
          lenght_t=120,
          B1=B1,
          B2_start=B2_start,
          B2=B2,
          v_to_r_ratio=v_to_r_ratio,
          w_to_r_ratio=w_to_r_ratio,
          w_to_v_ratio=w_to_v_ratio,
          m_to_r= m_to_r,
          m_v50=m_v50,
          v_v50=v_v50,
          w_v50=w_v50,
          testing_ratio_symp_1=testing_ratio_symp_1,
          testing_ratio_symp_2_start=testing_ratio_symp_2_start,
          testing_ratio_symp_2=testing_ratio_symp_2,
          Beta_other=Beta_other
          )
  

#st.write(population['population'].iloc[0])


multiplier = 1

divider = int(population['population'].iloc[0]/10)

if Y_Axis == 'per Population':
     multiplier = 1
elif Y_Axis == 'Count':
     multiplier = int(population['population'].iloc[0]/10)







################################### actual data  ##############################
###############################################################################


ITL_OWD = get_FR_with_testing(OWD=OWD,country=selected_country)
Number_of_people_with_any_Symptoms = ITL_OWD['new_tests'].values
Number_of_other_diseases =(Number_of_people_with_any_Symptoms - ITL_OWD['new_cases'])

# st.write(ITL_OWD.head(10))
#####################################  Active cases ##########################
##############################################################################

new_cases = ITL_OWD['new_tests'].values



#new_cases=ITL_OWD['new_cases']

#active_cases.plot()
#st.pyplot()


################################### PLOTS  ##############################
###############################################################################

if map_type == 'Subplots':

     f,axs=plt.subplots(3,2,figsize=(10,7),constrained_layout=True)
     ax=axs.ravel()
     M=np.array(M)
     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     ########## Active Cases ######### plot [0]
     ### Simulation
     ax[0].plot( (multiplier) * M.sum(axis=0)/1000,label='Active Mild')          ## Active V
     ax[0].plot( (multiplier) * W.sum(axis=0)/1000,label='Active Severe')        ## Active V
     ax[0].plot( (multiplier) * V.sum(axis=0)/1000,label='Active Ventilator')    ## Active V
     ax[0].plot( (multiplier) * (M.sum(axis=0) + V.sum(axis=0) + W.sum(axis=0))/1000,label='All Active') ## Sum of active M,W,V
     ax[0].axvline(B2_start, color='r', linestyle='dashed',lw=.5) ## vertical line (day Beta changed)
     ### actual
     #ax[0].plot(active_cases,color='k',label='Actual active cases') ## active cases rolling 8 new cases
     ax[0].set_title('Active')
     ax[0].legend()
     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     ######### Recovered Cases ######### plot [1]
     R=np.array(R)
     ax[1].plot((multiplier) * R/1000,label='Model')
     ax[1].set_title('Recovered')

     ax[1].plot((multiplier) * df_recovered.iloc[SHIFT:].values/divider,color='k',label='Actual') ## rcovered
     ax[1].legend()
     ######### Total Death ######### plot [2]

     D=np.array(D)
     ax[2].plot((multiplier) * D/1000,label='Model')
     ax[2].set_title('Death')

     ax[2].plot((multiplier) *  ITL_OWD['total_deaths'].iloc[SHIFT:].values/divider,color='k',label='Actual')   ## dead
     ax[2].legend()


     ######### Tested ######### plot [3]

     ax[3].plot((multiplier) *  Testing/1000)
     ax[3].axvline(testing_ratio_symp_2_start, color='r', linestyle='dashed',lw=.5)
     ax[3].set_title('Tested (t)')

     test_smooth = ITL_OWD['new_tests'].iloc[SHIFT:].rolling(1).mean().fillna(0).astype('int')

     ax[3].plot((multiplier) * test_smooth.values/divider,color = 'k')

     ######### New Daily Cases ######### plot [4]

     ax[4].plot((multiplier) *  M[1,:]/1000, label='MILD')   
     ax[4].plot((multiplier) *  confirmed_daily/1000,label=r'$\beta \times M_{t-1}$')   
     
     ax[4].set_title('Daily New Cases')
     ax[4].plot((multiplier) * ITL_OWD['new_cases'].iloc[SHIFT:].values/6e6,color='k',label='Actual data confirmed cases')   ## New Daily Cases
     ax[4].legend()
    #ax[4].legend()

     ######### Comparison ######### plot [5]
     severe_testing = W[0,:]
     sympt_testing = Testing - W[0,:]
     sympt_testing_negative = sympt_testing - M[1,:]
     sympt_testing_positive = M[1,:]


     ax[5].plot((multiplier) * severe_testing/1000,label = 'severe_testing')
     ax[5].plot((multiplier) * sympt_testing/1000,label = 'sympt_testing')
     ax[5].plot((multiplier) * sympt_testing_negative/1000 ,label = 'sympt_testing_negative ')
     ax[5].plot((multiplier) * sympt_testing_positive/1000,label = 'sympt_testing_positive')

     ax[5].legend()


     ax[5].set_title('Comparison')
     st.pyplot()
else:
     M=np.array(M)
     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     ########## Active Cases ######### plot [0]
     ### Simulation
     plt.plot(M.sum(axis=0)/1000,label='Active Mild')          ## Active V
     plt.plot(W.sum(axis=0)/1000,label='Active Severe')        ## Active V
     plt.plot(V.sum(axis=0)/1000,label='Active Ventilator')    ## Active V
     plt.plot((M.sum(axis=0) + V.sum(axis=0) + W.sum(axis=0))/1000,label='All Active') ## Sum of active M,W,V
     plt.axvline(B2_start, color='r', linestyle='dashed',lw=.5) ## vertical line (day Beta changed)
     ### actual
     
     plt.title('Active')
     plt.legend()

     st.pyplot()

     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     ######### Recovered Cases ######### plot [1]
     R=np.array(R)
     plt.plot(R/1000,label='Model')
     plt.title('Recovered')

     plt.plot(df_recovered.iloc[SHIFT:].values/divider,color='k',label='Actual') ## rcovered
     plt.legend()

     st.pyplot()

     ######### Total Death ######### plot [2]

     D=np.array(D)
     plt.plot(D/1000,label='Model')
     plt.title('Death')

     plt.plot(ITL_OWD['total_deaths'].iloc[SHIFT:].values/divider,color='k',label='Actual')   ## dead
     plt.legend()

     st.pyplot()

     ######### Tested ######### plot [3]

     plt.plot(Testing/1000)
     plt.axvline(testing_ratio_symp_2_start, color='r', linestyle='dashed',lw=.5)
     plt.title('Tested (t)')
     test_smooth = ITL_OWD['new_tests'].iloc[SHIFT:].rolling(1).mean().fillna(0).astype('int')
     plt.plot(test_smooth.values/6e6,color = 'k')
     st.pyplot()
     ######### New Daily Cases ######### plot [4]

     plt.plot(M[1,:]/1000)   
     plt.title('Daily New Cases')
     plt.plot(ITL_OWD['new_cases'].iloc[SHIFT:].values/divider,color='k')   ## New Daily Cases
     st.pyplot()

     ######### Comparison ######### plot [5]
     severe_testing = W[0,:]
     sympt_testing = Testing - W[0,:]
     sympt_testing_negative = sympt_testing - M[1,:]
     sympt_testing_positive = M[1,:]


     plt.plot(severe_testing/1000,label = 'severe_testing')
     plt.plot(sympt_testing/1000,label = 'sympt_testing')
     plt.plot(sympt_testing_negative/1000 ,label = 'sympt_testing_negative ')
     plt.plot(sympt_testing_positive/1000,label = 'sympt_testing_positive')

     plt.legend()


     plt.title('Comparison')
     st.pyplot()