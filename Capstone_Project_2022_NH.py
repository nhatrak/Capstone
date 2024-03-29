"""
Natalie Hatrak

Data Science Academy Capstone Project

January 2022
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
st.set_page_config(layout="wide")
from scipy.stats import ttest_ind
from scipy.stats import f_oneway 
from scipy.stats import chi2_contingency
import plotly.graph_objects as go

# Read in data
df = pd.read_csv("data_capstone_dsa2021_2022.csv")

# Cleaning up state column. First uppercase all for easier standardization
df['state'] = df['state'].str.upper()

# Create a dictionary  of all state names and abbreviations
us_state_to_abbrev = {
	    "ALABAMA": "AL",
	    "ALASKA": "AK",
	    "ARIZONA": "AZ",
	    "ARKANSAS": "AR",
	    "CALIFORNIA": "CA",
	    "COLORADO": "CO",
	    "CONNECTICUT": "CT",
	    "DELAWARE": "DE",
	    "FLORIDA": "FL",
	    "GEORGIA": "GA",
	    "HAWAII": "HI",
	    "IDAHO": "ID",
	    "ILLINOIS": "IL",
	    "INDIANA": "IN",
	    "IOWA": "IA",
	    "KANSAS": "KS",
	    "KENTUCKY": "KY",
	    "LOUISIANA": "LA",
	    "MAINE": "ME",
	    "MARYLAND": "MD",
	    "MASSACHUSETTS": "MA",
	    "MICHIGAN": "MI",
	    "MINNESOTA": "MN",
	    "MISSISSIPPI": "MS",
	    "MISSOURI": "MO",
	    "MONTANA": "MT",
	    "NEBRASKA": "NE",
	    "NEVADA": "NV",
	    "NEW HAMPSHIRE": "NH",
	    "NEW JERSEY": "NJ",
	    "NEW MEXICO": "NM",
	    "NEW YORK": "NY",
	    "NORTH CAROLINA": "NC",
	    "NORTH DAKOTA": "ND",
	    "OHIO": "OH",
	    "OKLAHOMA": "OK",
	    "OREGON": "OR",
	    "PENNSYLVANIA": "PA",
	    "RHODE ISLAND": "RI",
	    "SOUTH CAROLINA": "SC",
	    "SOUTH DAKOTA": "SD",
	    "TENNESSEE": "TN",
	    "TEXAS": "TX",
	    "UTAH": "UT",
	    "VERMONT": "VT",
	    "VIRGINIA": "VA",
	    "WASHINGTON": "WA",
	    "WEST VIRGINIA": "WV",
	    "WISCONSIN": "WI",
	    "WYOMING": "WY",
        "PUERTO RICO": "PR",
	}

# get rid of special characters to start the cleaning
remove_characters = [".", ",", "(", ")"]

for character in remove_characters:
    df['state'] = df['state'].str.replace(character, "", regex=True)


# get rid of USA and UNITED STATES and turn slash and double space into single space
char_to_replace = {'/': ' ',
                   ' USA': '',
                   'USA ': '',
                   'USA': '',
                   ' UNITED STATES OF AMERICA': '',
                   ' UNITED STATES': '',
                   'UNITED STATES ': '',
                   ' US': '',
                   '  ': ' ',
                   }

for key, value in char_to_replace.items():
    df['state'] = df['state'].str.replace(key, value)



# Get abbreviations for already clean data (ie contains only state name)
df['temp1'] = df['state'].map(us_state_to_abbrev)
df = df.replace(np.nan, '', regex=True)


# Find any observations where the state listed is already the abrreviation
abbrevs= list(us_state_to_abbrev.values()) # turn values from state/abrrev dictionary into a list  
df['temp2'] = np.where(df['state'].isin(abbrevs), df['state'], "")


# combine anywhere where we now have state so far
df['state2']= df["temp1"] + df["temp2"]
df.drop(columns=['temp1', 'temp2'], inplace=True)


# Separate data into that which mapped successfully already and that which did not
df_good_state = df.loc[df['state2'] != ""]
df_no_state = df.loc[df['state2'] == ""]
df_no_state.drop(columns=['state2'], inplace=True) # since blank, dont need it and will get in the way later


# Look for observations where the state or abbrev is present within the string but may contain other words as well.
# Split state up by each individual word 
df_no_state_expand= df_no_state['state'].str.split(' ', expand=True)
# Rename columns so can use them later 
df_no_state_expand.columns = ['var1', 'var2', 'var3','var4', 'var5', 'var6','var7', 'var8', 'var9','var10', 'var11']


state_list= list(us_state_to_abbrev.keys()) # turn keys from state/abrrev dictionary into a list   

# Compare var1-11 to the state and abbrev lists and output anywhere that they match
for i in range(1,12):
    df_no_state_expand['find_state%s'%i] = np.where(df_no_state_expand['var%s'%i].isin(state_list), df_no_state_expand['var%s'%i], "")
    df_no_state_expand['find_abbrev%s'%i] = np.where(df_no_state_expand['var%s'%i].isin(abbrevs), df_no_state_expand['var%s'%i], "")
    
# collapse all of the find_state and Find_abbrev vars     
df_no_state_expand['state2_temp']=df_no_state_expand["find_state1"] + df_no_state_expand["find_state2"] \
    + df_no_state_expand["find_state3"] + df_no_state_expand["find_state4"] + df_no_state_expand["find_state5"] \
    + df_no_state_expand["find_state6"] + df_no_state_expand["find_state7"] + df_no_state_expand["find_state8"] \
    + df_no_state_expand["find_state9"] + df_no_state_expand["find_state10"] + df_no_state_expand["find_state11"]     

df_no_state_expand['abbrev_temp']=df_no_state_expand["find_abbrev1"] + df_no_state_expand["find_abbrev2"] \
    + df_no_state_expand["find_abbrev3"] + df_no_state_expand["find_abbrev4"] + df_no_state_expand["find_abbrev5"] \
    + df_no_state_expand["find_abbrev6"] + df_no_state_expand["find_abbrev7"] + df_no_state_expand["find_abbrev8"] \
    + df_no_state_expand["find_abbrev9"] + df_no_state_expand["find_abbrev10"] + df_no_state_expand["find_abbrev11"]     

    
# map those with full state name to original state dictionary to get the abbreviation 
df_no_state_expand['temp3'] = df_no_state_expand['state2_temp'].map(us_state_to_abbrev)
df_no_state_expand = df_no_state_expand.replace(np.nan, '', regex=True)

# combine anywhere that we now have state 
df_no_state_expand['state2'] = df_no_state_expand['abbrev_temp']+ df_no_state_expand['temp3']
df_no_state_expand=df_no_state_expand[['state2']] # only keep this var bc it is the only one we care about

#merge the new variable in with the original no_state dataframe (can merge on index solely)
df_no_state=df_no_state.merge(df_no_state_expand,left_index=True, right_index=True)

#combine the new no_state data that mostly has state2 filled in with the good_state data to get the final dataset. 
final_dataset= pd.concat([df_good_state, df_no_state])

#put blank to Unknown State
final_dataset['state_final'] = np.where(final_dataset['state2'].isin(abbrevs), final_dataset['state2'], "Unknown_State")

#add age buckets
def age_buckets(x): 
    if x < 30: return ' Under 30'
    elif x < 40: return '30-39' 
    elif x < 50: return '40-49' 
    elif x < 60: return '50-59' 
    elif x >=60: return '60+' 
    else: return 'other'
final_dataset['Age_Range']=final_dataset.age.apply(age_buckets)

# =============================================================================
# All code above is reading in data and doing cleaning on state
# Below code has all visualizations and code for displaying in streamlit(st.xxxxxx)
# =============================================================================

st.title("Capstone Project - January 2022")

Title1 = '<p style="font-family:sans-serif; text-align: center; color:Blue; font-size: 20px;">This dashboard was produced for a capstone project of Data Science 2021-2022.  <br> \
            By: Natalie Hatrak, PAR - K12 <br> Observations in data:  N=1169</p>'
st.markdown(Title1, unsafe_allow_html=True)
Title2 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;">Breakdown of Demographic Variables</p>'
st.markdown(Title2, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# DEMOGRAPHIC INFO
# =============================================================================
#add pie charts for demographic information
#Gender
Gender_Pie = final_dataset['gender'].value_counts()
Gender_Pie2=pd.DataFrame(Gender_Pie)
Gender_Pie2.reset_index(inplace=True)
Gender_Pie2 = Gender_Pie2.rename(columns = {'index':'Gender','gender':'N'})
Gender_Pie2.sort_values(by=['Gender'], inplace=True)
Pie1=px.pie(Gender_Pie2, values='N',names='Gender',title='Gender Percentage')
Pie1.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Pie1.update_traces(textinfo='label+percent')
Pie1.update_traces(sort=False) 

#States
State_Pie = final_dataset['state_final'].value_counts()
State_Pie2=pd.DataFrame(State_Pie)
State_Pie2.reset_index(inplace=True)
State_Pie2 = State_Pie2.rename(columns = {'index':'State','state_final':'N'}).sort_values('N', ascending = False)
State_Pie2_Top=State_Pie2[:11].copy()
new_row = pd.DataFrame(data = {'State':['States less than N=30'],'N' : [State_Pie2['N'][11:].sum()]})
State_Pie2_Top.sort_values(by=['State'], inplace=True)
State_Pie3=pd.concat([State_Pie2_Top, new_row],ignore_index=True)
Pie2=px.pie(State_Pie3, values='N',names='State',labels='State', title='State Percentage')
Pie2.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Pie2.update_traces(textinfo='label+percent')
Pie2.update_traces(sort=False) 

#home_computer
Comp_Pie = final_dataset['home_computer'].value_counts()
Comp_Pie2=pd.DataFrame(Comp_Pie)
Comp_Pie2.reset_index(inplace=True)
Comp_Pie2 = Comp_Pie2.rename(columns = {'index':'Home_computer','home_computer':'N'})
Comp_Pie2.sort_values(by=['Home_computer'], inplace=True)
Pie3=px.pie(Comp_Pie2, values='N',names='Home_computer',title='Home Computer Percentage')
Pie3.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Pie3.update_traces(textinfo='label+percent')
Pie3.update_traces(sort=False) 

#age bucket
Age_Pie = final_dataset['Age_Range'].value_counts()
Age_Pie2=pd.DataFrame(Age_Pie)
Age_Pie2.reset_index(inplace=True)
Age_Pie2 = Age_Pie2.rename(columns = {'index':'Age_Range','Age_Range':'N'})
Age_Pie2.sort_values(by=['Age_Range'], inplace=True)
Pie4=px.pie(Age_Pie2, values='N',names='Age_Range',title='Age Range Percentage')
Pie4.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Pie4.update_traces(textinfo='label+percent')
Pie4.update_traces(sort=False) 

#produce 2 columns and have Gender/Home_computer side by side and age/states side by side   
col1, col2=st.columns([5,5])
with col1:
    st.plotly_chart (Pie1, use_container_width=True)
with col2:
    st.plotly_chart(Pie3, use_container_width=True)
col3, col4=st.columns([5,5])
with col3:
    st.plotly_chart (Pie4, use_container_width=True)
with col4:
    st.plotly_chart(Pie2, use_container_width=True)

st.markdown("---")
# =============================================================================
# Gender and Sum_Score
# =============================================================================
gender_means = final_dataset.groupby(['gender'])['sum_score'].mean().round(2).reset_index()
f_means = gender_means.loc[gender_means['gender'] == 'Female']
m_means = gender_means.loc[gender_means['gender'] == 'Male']
gender_means2=f_means.append(m_means)
fig1=px.bar(gender_means2,x='gender',y='sum_score',color='gender',title ="Total Score Mean by Gender", 
            labels={'gender': 'Gender', 'sum_score':'Total Score Mean'})
fig1.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

Gender_Score_Box=px.box(final_dataset, x='gender', y='sum_score', color='gender',  title="Box plot of Total Score by Gender",
            category_orders = {"gender" : ["Female", "Male"]}, labels={'gender': 'Gender', 'sum_score':'Total Score'})
Gender_Score_Box.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

t_stat, p = ttest_ind(final_dataset.query('gender=="Male"')['sum_score'], final_dataset.query('gender=="Female"')['sum_score'])

Title3 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;">Examining Gender and Total Score</p>'
st.markdown(Title3, unsafe_allow_html=True)

col5, col6=st.columns(2)
with col5:
    st.plotly_chart(fig1, use_container_width=True)
with col6:
    st.plotly_chart(Gender_Score_Box)
    
st.write("**Is there a difference in sum_score between Male and Female?**")
st.write("Ho: There is no difference  \n \
         H1: There is a difference")
st.write("T_test - p_value: ",round(p,4))
st.write("The p-value is less than 0.05, we reject the null hypothesis.  \n \
             There is a difference in sum_score between Male and Female")
             
st.markdown("---")
# =============================================================================
# Gender and Total Time
# =============================================================================
gender_means_time = final_dataset.groupby(['gender'])['rt_total'].mean().round(2).reset_index()
f_means_time = gender_means_time.loc[gender_means_time['gender'] == 'Female']
m_means_time = gender_means_time.loc[gender_means_time['gender'] == 'Male']
gender_means_time2=f_means_time.append(m_means_time)
fig2=px.bar(gender_means_time2,x='gender',y='rt_total',color='gender',title ="Total Time Mean by Gender",
            labels={'gender': 'Gender', 'rt_total':'Total Time Mean'})
fig2.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

Gender_Time_Box=px.box(final_dataset, x='gender', y='rt_total', color='gender',  title="Box plot of Total Time by Gender",
            category_orders = {"gender" : ["Female", "Male"]}, labels={'gender': 'Gender', 'rt_total':'Total Time'})
Gender_Time_Box.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

t_stat2, p2 = ttest_ind(final_dataset.query('gender=="Male"')['rt_total'], final_dataset.query('gender=="Female"')['rt_total'])

Title4 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;">Examining Gender and Total Time</p>'
st.markdown(Title4, unsafe_allow_html=True)  
col7, col8=st.columns(2)
with col7:
    st.plotly_chart(fig2, use_container_width=True)
with col8:
    st.plotly_chart(Gender_Time_Box)
    
st.write("**Is there a difference in total time between Male and Female?**")
st.write("Ho: There is no difference  \n \
            H1: There is a difference")
st.write("T_test - p_value: ",round(p2,4))
st.write("The p-value is less than 0.05, we reject the null hypothesis.  \n \
             There is a difference in total time between Male and Female")

st.markdown("---")
# =============================================================================
# Age_Group and Sum_Score and total time
# =============================================================================
age_means = final_dataset.groupby(['gender'])['age'].mean().round(2).reset_index()
f_means_age = age_means.loc[age_means['gender'] == 'Female']
m_means_age = age_means.loc[age_means['gender'] == 'Male']
age_meanss2=f_means_age.append(m_means_age)
fig11=px.bar(age_meanss2,x='gender',y='age',color='gender',title ="Age Mean by Gender", 
            labels={'gender': 'Gender', 'age':'Age Mean'})
fig11.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

Gender_Age_Box=px.box(final_dataset, x='gender', y='age', color='gender',  title="Box plot of Age by Gender",
            category_orders = {"gender" : ["Female", "Male"]}, labels={'gender': 'Gender', 'age':'Age'})
Gender_Age_Box.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

#get totals with no age range to add on to age_range graphs
Total_scr_for_graphs = pd.DataFrame()
Total_scr_for_graphs2=gender_means2.append(Total_scr_for_graphs)
Total_scr_for_graphs2['Age_Range'] = "ALL"
age_gender_scr_means = final_dataset.groupby(['gender', 'Age_Range'])['sum_score'].mean().round(2).reset_index()
age_gender_scr_means2=age_gender_scr_means.append(Total_scr_for_graphs2)
Gender_Age_Scr_Box=px.scatter(age_gender_scr_means2, x='Age_Range', y='sum_score', color='gender',  
            title="Total Score Mean by Age Group and Gender",
            category_orders = {"Age_Range" : [" Under 30", "30-39", "40-49", "50-59", "60+", "ALL"]}, 
            labels={'gender': 'Gender', 'sum_score':'Total Score Mean', 'Age_Range': 'Age Group'}, text='sum_score')
Gender_Age_Scr_Box.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Gender_Age_Scr_Box.update_traces(textposition='middle right')

#get totals with no age range to add on to age_range graphs
Total_times_for_graphs = pd.DataFrame()
Total_times_for_graphs2=gender_means_time2.append(Total_times_for_graphs)
Total_times_for_graphs2['Age_Range'] = "ALL"
age_gender_time_means = final_dataset.groupby(['gender', 'Age_Range'])['rt_total'].mean().round(2).reset_index()
age_gender_time_means2=age_gender_time_means.append(Total_times_for_graphs2)
Gender_Age_time_Box=px.scatter(age_gender_time_means2, x='Age_Range', y='rt_total', color='gender',  
            title="Total Time Mean by Age Group and Gender", 
            category_orders = {"Age_Range" : [" Under 30", "30-39", "40-49", "50-59", "60+", "ALL"]}, 
            labels={'gender': 'Gender', 'rt_total':'Total Time Mean', 'Age_Range': 'Age Group'}, text='rt_total')
Gender_Age_time_Box.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Gender_Age_time_Box.update_traces(textposition='middle right')


Title5 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;">Examining Gender and Age</p>'
st.markdown(Title5, unsafe_allow_html=True)

col9, col10=st.columns(2)
with col9:
    st.plotly_chart(fig11, use_container_width=True)
with col10:
    st.plotly_chart(Gender_Age_Box)

df_anova = final_dataset[['sum_score','Age_Range']]
grps = pd.unique(final_dataset.Age_Range.values)
d_data = {grp:df_anova['sum_score'][df_anova.Age_Range == grp] for grp in grps}
F3, p3 = f_oneway(d_data[" Under 30"], d_data["30-39"], d_data["40-49"],d_data["50-59"],d_data["60+"])

df_anova2 = final_dataset[['rt_total','Age_Range']]
grps2 = pd.unique(final_dataset.Age_Range.values)
d_data2 = {grp:df_anova2['rt_total'][df_anova2.Age_Range == grp] for grp in grps2}
F4, p4 = f_oneway(d_data2[" Under 30"], d_data2["30-39"], d_data2["40-49"],d_data2["50-59"],d_data2["60+"])


col11, col12=st.columns(2)
with col11:
    st.write("**Is there a difference in total scores between the Age Groups?**")
    st.write("Ho: There is no difference  \n \
            H1: There is a difference")
    st.write("Anova - F statistic: ",round(F3,4))
    st.write("Anova - p_value: ",round(p3,4))
    st.write("The p-value is less than 0.05, we reject the null hypothesis.  \n \
             There is a difference in total scores between the Age Groups")
with col12:
    st.write("**Is there a difference in total times between the Age Groups?**")
    st.write("Ho: There is no difference  \n \
            H1: There is a difference")
    st.write("Anova - F statistic: ",round(F4,4))
    st.write("Anova - p_value: ",round(p4,4))
    st.write("The p-value is less than 0.05, we reject the null hypothesis.  \n \
             There is a difference in total times between the Age Groups")
    
col13, col14=st.columns(2)
with col13:
    st.plotly_chart(Gender_Age_Scr_Box, use_container_width=True)
with col14:
    st.plotly_chart(Gender_Age_time_Box, use_container_width=True)

st.markdown("---")   
# =============================================================================
# STATES and Sum_Score and Total Time
# =============================================================================
Title6 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;">Examining States Total Score Mean and Total Time Mean by Gender</p>'
st.markdown(Title6, unsafe_allow_html=True) 
get_states=final_dataset['state_final'].drop_duplicates()
get_states.sort_values(ascending=True, inplace=True)
option = st.selectbox('Select a State to See Mean Scores by Gender and Mean Total Times by Gender',get_states)
#sum_score
state_gender_means = final_dataset.groupby(['gender', 'state_final'])['sum_score'].mean().reset_index()
state_f_means = state_gender_means.loc[state_gender_means['gender'] == 'Female']
state_m_means = state_gender_means.loc[state_gender_means['gender'] == 'Male']
state_gender_means2=state_f_means.append(state_m_means)
state_gender_means2_query=state_gender_means2.loc[state_gender_means2['state_final'] == option]
fig3=px.bar(state_gender_means2_query,x='gender',y='sum_score',color='gender',title ="Total Score Mean by Gender and State",
            labels={'gender': 'Gender', 'sum_score':'Total Score Mean'})
fig3.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
#total time
time_gender_means = final_dataset.groupby(['gender', 'state_final'])['rt_total'].mean().reset_index()
time_f_means = time_gender_means.loc[time_gender_means['gender'] == 'Female']
time_m_means = time_gender_means.loc[time_gender_means['gender'] == 'Male']
time_gender_means2=time_f_means.append(time_m_means)
time_gender_means2_query=time_gender_means2.loc[time_gender_means2['state_final'] == option]
fig4=px.bar(time_gender_means2_query,x='gender',y='rt_total',color='gender',title ="Total Time Mean by Gender and State",
            labels={'gender': 'Gender', 'rt_total':'Total Time Mean'})
fig4.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

col15, col16=st.columns(2)
with col15:
    st.plotly_chart(fig3, use_container_width=True)
with col16:
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")
# =============================================================================
# Item Correlation
# =============================================================================
Title7 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;">Examining Item Scores and Total Score Correlation</p>'
st.markdown(Title7, unsafe_allow_html=True)  
Title8 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 15px;">Looking at sum_score row, you can see gs_4, gs_7, gs_10, and gs_12 are the only items over .50. <br> \
            gs_7 has the highest correlations with sum_score at .58 </p>'
st.markdown(Title8, unsafe_allow_html=True)  

#Get just item scores and sum_score for correlation heat matrix
final_dataset2=final_dataset.iloc[:, 20:] 
final_dataset2.drop(columns=['state2', 'gender', 'home_computer', 'state', 'age', 'state_final', 'Age_Range'], inplace=True) 
corrs=final_dataset2.corr(method ='pearson')
mask=np.zeros_like(corrs)
mask[np.triu_indices_from(mask)] = True

fig=plt.figure(figsize=(7, 3))
sb.set(font_scale=.3)
heatmap=sb.heatmap(corrs, 
            vmin=-1,
            vmax=1,
            xticklabels=corrs.columns,
            yticklabels=corrs.columns,
            cmap='RdBu_r',
            mask=mask,
            annot=True,
            fmt='.2f',
            linewidth=2)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':10}, pad=12)
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=4, rotation=30)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=4, rotation=0)
st.pyplot(fig)

st.markdown("---")
# =============================================================================
# Item Statistics
# =============================================================================
#produce table of item stats (like IA)
for i in range(1,21):
    datax = final_dataset['gs_%s'%i].value_counts()
    datay = pd.DataFrame({'gs_%s'%i: datax.index,'Frequency': datax.values,'Percent': ((datax.values/datax.values.sum())*100).round(2)})
    datay.sort_values(by=['gs_%s'%i], inplace=True)
    mn=final_dataset.groupby(by="gs_%s"%i,as_index=False).mean('sum_score').round(2)
    mn = mn.rename(columns={'sum_score': 'Total_Score_Mean'})
    mn=mn[['gs_%s'%i,'Total_Score_Mean']]
    std=final_dataset.groupby(["gs_%s"%i])['sum_score'].std().round(2)
    std2=pd.DataFrame(std)
    std2 = std2.rename(columns={'sum_score': 'Total_Score_STD'})
    mn_std=mn.merge(std2,left_index=True, right_index=True)
    
    mn_time=final_dataset.groupby(by="gs_%s"%i,as_index=False).mean('rt_total').round(2)
    mn_time = mn_time.rename(columns={'rt_total': 'Total_Time_Mean'})
    mn_time=mn_time[['gs_%s'%i,'Total_Time_Mean']]
    std_time=final_dataset.groupby(["gs_%s"%i])['rt_total'].std().round(2)
    std2=pd.DataFrame(std_time)
    std_time2 = std2.rename(columns={'rt_total': 'Total_Time_STD'})
    mn_std_time=mn_time.merge(std_time2,left_index=True, right_index=True)
    time_scores=mn_std.merge(mn_std_time,left_on='gs_%s'%i, right_on='gs_%s'%i)
    
    freq_mn_std=datay.merge(time_scores,left_on='gs_%s'%i, right_on='gs_%s'%i)
    freq_mn_std['Item'] = "gs_%s"%i
    freq_mn_std = freq_mn_std.rename(columns={'gs_%s'%i: 'Item_Score'})
    freq_mn_std.sort_values(by=['Item_Score'], inplace=True)
    if i == 1:
        all_freq_mn_std=freq_mn_std
    else:
        all_freq_mn_std=all_freq_mn_std.append(freq_mn_std)

#get pvalues and corrs in same table
for j in range(1,21):
    datax2 = final_dataset['gs_%s'%j].value_counts()
    datay2 = pd.DataFrame({'gs_%s'%j: datax.index,'Pvalue': ((datax2.values/datax2.values.sum())).round(2)})
    datay2=datay2.loc[datay2['gs_%s'%j] == 1]
    datay2['Item'] = "gs_%s"%j
    datay2 = datay2.rename(columns={'gs_%s'%j: 'Item_Score'})
    if j == 1:
        all_pval=datay2
    else:
        all_pval=all_pval.append(datay2)

corrs_only=corrs.iloc[-1]
corrs_only2=corrs_only.to_frame()
corrs_only2.reset_index(inplace=True)
corrs_only2=corrs_only2.rename(columns = {'index':'Item', 'sum_score':'Correlation'})
p_val_corr=all_pval.merge(corrs_only2,left_on='Item', right_on='Item')
p_val_corr.drop(columns=['Item_Score'], inplace=True) 
p_val_corr=p_val_corr[["Item","Pvalue",'Correlation']]
#get table of stats to print
Title9 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;">Examining Item Statistics</p>'
st.markdown(Title9, unsafe_allow_html=True) 
# =============================================================================
Item_Box=px.box(p_val_corr, y='Pvalue', points='all', hover_data=["Item", "Pvalue"], title="Box plot of Pvalue")
Item_Box.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Item_Box2=px.box(p_val_corr, y='Correlation', points='all', hover_data=["Item", "Correlation"], title="Box plot of Correlation")
Item_Box2.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
col17, col18=st.columns(2)
with col17:
    st.plotly_chart(Item_Box, use_container_width=True)
with col18:
    st.plotly_chart(Item_Box2, use_container_width=True)
# =============================================================================
#reorder columns
all_freq_mn_std = all_freq_mn_std[["Item", "Item_Score", "Frequency", "Percent", "Total_Score_Mean", "Total_Score_STD",
                                   "Total_Time_Mean", "Total_Time_STD"]]
get_items=all_freq_mn_std['Item'].drop_duplicates()
get_items.sort_values(ascending=True, inplace=True)
get_item_list=get_items.tolist()

get_item_list2=[]
get_item_list2=get_item_list[:]
get_item_list2.append('Select all Items')
get_item_dropdown=st.multiselect('Select Individual Items or Choose Select All Items to See Item Statistics',get_item_list2)
if 'Select all Items' in get_item_dropdown:
 	get_item_dropdown=get_item_list
get_item_dropdown=get_item_dropdown
option_item_query = all_freq_mn_std.loc[all_freq_mn_std["Item"].isin(get_item_dropdown)]
option_item_query2 = p_val_corr.loc[p_val_corr["Item"].isin(get_item_dropdown)]
#below code is to hide row indices
#CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
st.markdown('## Table of Item Statistics by Item Score')
st.table(option_item_query.style.set_precision(2))
st.markdown('## Table of Overall Item Pvalue and Correlations')
st.table(option_item_query2.style.set_precision(2))

st.markdown("---")
# =============================================================================
# Total Times Means by Overall and Demographics
# =============================================================================
Title10 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;"> Examining Item Score for Overall, Gender, Age, and Home Computer </p>'
            
st.markdown(Title10, unsafe_allow_html=True) 
for k in range(1,21):
    time_means = final_dataset.groupby(['gs_%s'%k,'home_computer'])['rt_total'].mean().round(2).reset_index()
    time_means['Item'] = "gs_%s"%k
    time_means = time_means.rename(columns={'gs_%s'%k: 'Item_Score'})
    time_means['Answered'] = np.where(time_means['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    time_means2 = final_dataset.groupby(['gs_%s'%k,'gender'])['rt_total'].mean().round(2).reset_index()
    time_means2['Item'] = "gs_%s"%k
    time_means2 = time_means2.rename(columns={'gs_%s'%k: 'Item_Score'})
    time_means2['Answered'] = np.where(time_means2['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    time_means3 = final_dataset.groupby(['gs_%s'%k,'Age_Range'])['rt_total'].mean().round(2).reset_index()
    time_means3['Item'] = "gs_%s"%k
    time_means3 = time_means3.rename(columns={'gs_%s'%k: 'Item_Score'})
    time_means3['Answered'] = np.where(time_means3['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    time_means4 = final_dataset.groupby(['gs_%s'%k])['rt_total'].mean().round(2).reset_index()
    time_means4['Item'] = "gs_%s"%k
    time_means4 = time_means4.rename(columns={'gs_%s'%k: 'Item_Score'})
    time_means4['Answered'] = np.where(time_means4['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    if k == 1:
        Time_HomeComputer=time_means
        Time_Gender=time_means2
        Time_Age=time_means3
        Time_Total=time_means4
    else:
        Time_HomeComputer=Time_HomeComputer.append(time_means)
        Time_Gender=Time_Gender.append(time_means2)
        Time_Age=Time_Age.append(time_means3)
        Time_Total=Time_Total.append(time_means4)
        
Scatter_Time_Total = px.scatter(Time_Total, x='Item', y='rt_total', color='Answered',
                    title="Total Time Mean by Item Score - Overall",
                    labels={'rt_total':'Total Time Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_Time_Total.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_Time_Total.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

Scatter_Time_Gender = px.scatter(Time_Gender, x='Item', y='rt_total', color='Answered', facet_col=('gender'),
                    title="Total Time Mean by Item Score - Gender",
                    labels={'gender': 'Gender', 'rt_total':'Total Time Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_Time_Gender.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_Time_Gender.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

Scatter_Time_Age = px.scatter(Time_Age, x='Item', y='rt_total', color='Answered', facet_col=('Age_Range'),
                    facet_col_wrap=3, title="Total Time Mean by Item Score - Age",
                    labels={'Age_Range': 'Home Age_Range', 'rt_total':'Total Time Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_Time_Age.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_Time_Age.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

Scatter_Time_HomeComputer = px.scatter(Time_HomeComputer, x='Item', y='rt_total', color='Answered', facet_col=('home_computer'),
                    title="Total Time Mean by Item Score - Home Computer",
                    labels={'home_computer': 'Home Computer', 'rt_total':'Total Time Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_Time_HomeComputer.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_Time_HomeComputer.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

# =============================================================================
# Total Score Means by Overall and Demographics
# =============================================================================
for k in range(1,21):
    scr_means = final_dataset.groupby(['gs_%s'%k,'home_computer'])['sum_score'].mean().round(2).reset_index()
    scr_means['Item'] = "gs_%s"%k
    scr_means = scr_means.rename(columns={'gs_%s'%k: 'Item_Score'})
    scr_means['Answered'] = np.where(scr_means['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    scr_means2 = final_dataset.groupby(['gs_%s'%k,'gender'])['sum_score'].mean().round(2).reset_index()
    scr_means2['Item'] = "gs_%s"%k
    scr_means2 = scr_means2.rename(columns={'gs_%s'%k: 'Item_Score'})
    scr_means2['Answered'] = np.where(scr_means2['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    scr_means3 = final_dataset.groupby(['gs_%s'%k,'Age_Range'])['sum_score'].mean().round(2).reset_index()
    scr_means3['Item'] = "gs_%s"%k
    scr_means3 = scr_means3.rename(columns={'gs_%s'%k: 'Item_Score'})
    scr_means3['Answered'] = np.where(scr_means3['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    scr_means4 = final_dataset.groupby(['gs_%s'%k])['sum_score'].mean().round(2).reset_index()
    scr_means4['Item'] = "gs_%s"%k
    scr_means4 = scr_means4.rename(columns={'gs_%s'%k: 'Item_Score'})
    scr_means4['Answered'] = np.where(scr_means4['Item_Score']== 1, 'Answered_Right', 'Answered_Wrong')
    if k == 1:
        scr_HomeComputer=scr_means
        scr_Gender=scr_means2
        scr_Age=scr_means3
        scr_Total=scr_means4
    else:
        scr_HomeComputer=scr_HomeComputer.append(scr_means)
        scr_Gender=scr_Gender.append(scr_means2)
        scr_Age=scr_Age.append(scr_means3)
        scr_Total=scr_Total.append(scr_means4)
        
Scatter_scr_Total = px.scatter(scr_Total, x='Item', y='sum_score', color='Answered',
                    title="Total Score Mean by Item Score - Overall",
                    labels={'sum_score':'Total Score Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_scr_Total.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_scr_Total.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

Scatter_scr_Gender = px.scatter(scr_Gender, x='Item', y='sum_score', color='Answered', facet_col=('gender'),
                    title="Total Score Mean by Item Score - Gender",
                    labels={'gender': 'Gender', 'sum_score':'Total Score Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_scr_Gender.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_scr_Gender.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

Scatter_scr_Age = px.scatter(scr_Age, x='Item', y='sum_score', color='Answered', facet_col=('Age_Range'),
                    facet_col_wrap=3, title="Total Score Mean by Item Score - Age",
                    labels={'Age_Range': 'Home Age_Range', 'sum_score':'Total Score Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_scr_Age.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_scr_Age.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

Scatter_scr_HomeComputer = px.scatter(scr_HomeComputer, x='Item', y='sum_score', color='Answered', facet_col=('home_computer'),
                    title="Total Score Mean by Item Score - Home Computer",
                    labels={'home_computer': 'Home Computer', 'sum_score':'Total Score Mean'},
                    category_orders = {"Item" : ["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"]},
                    height=500, width=900)
Scatter_scr_HomeComputer.update_layout(title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
Scatter_scr_HomeComputer.update_xaxes(tickvals=["gs_1", "gs_2", "gs_3", "gs_4", "gs_5", "gs_6","gs_7", "gs_8", "gs_9", "gs_10",
                                                 "gs_11", "gs_12", "gs_13", "gs_14", "gs_15", "gs_16","gs_17", "gs_18", "gs_19", "gs_20"])

# create radio button to select sum_score or rt_total
cola, colb,colc=st.columns(3)
with colb:
    option_scatters = st.radio('Select One to see Scatterplots',('Total Score', 'Total Time'))
    
col19, col20=st.columns(2)
with col19:
    if 'Total Score' in option_scatters: st.plotly_chart(Scatter_scr_Total, use_container_width=True)
    if 'Total Time' in option_scatters: st.plotly_chart(Scatter_Time_Total, use_container_width=True)
with col20:
    if 'Total Score' in option_scatters: st.plotly_chart(Scatter_scr_Gender, use_container_width=True)
    if 'Total Time' in option_scatters: st.plotly_chart(Scatter_Time_Gender, use_container_width=True)

col21, col22=st.columns(2)
with col21:
    if 'Total Score' in option_scatters: st.plotly_chart(Scatter_scr_Age, use_container_width=True)
    if 'Total Time' in option_scatters: st.plotly_chart(Scatter_Time_Age, use_container_width=True)
with col22:
    if 'Total Score' in option_scatters: st.plotly_chart(Scatter_scr_HomeComputer, use_container_width=True)
    if 'Total Time' in option_scatters: st.plotly_chart(Scatter_Time_HomeComputer, use_container_width=True)

st.markdown("---")

Title11 = '<p style="font-family:sans-serif; text-align: center; color:Green; font-size: 30px;"> Examining Total Score Means by Overall and Gender for Each US State </p>'
st.markdown(Title11, unsafe_allow_html=True) 


State_ForMap = final_dataset.groupby(["state_final"])["sum_score"].describe().round(2).reset_index()
State_ForMap1 = go.Figure(data=go.Choropleth(locations = State_ForMap['state_final'],z = State_ForMap['mean'],
locationmode = 'USA-states', # set of locations match entries in `locations`
colorscale = 'sunset',colorbar_title = "Total Score Mean"))
State_ForMap1.update_layout(dragmode = False,title_text = 'Total Score Mean by US State',geo_scope='usa',
                            title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

StateGen_ForMap = final_dataset.groupby(["state_final","gender"])["sum_score"].describe().round(2).reset_index()
StateGen_ForMapF = StateGen_ForMap.loc[StateGen_ForMap['gender'] == 'Female']
StateGen_ForMapM = StateGen_ForMap.loc[StateGen_ForMap['gender'] == 'Male']
StateGen_ForMap1_F = go.Figure(data=go.Choropleth(locations = StateGen_ForMapF['state_final'],z = StateGen_ForMapF['mean'],
locationmode = 'USA-states', # set of locations match entries in `locations`
colorscale = 'sunset',colorbar_title = "Total Score Mean"))
StateGen_ForMap1_F.update_layout(dragmode = False,title_text = 'Total Score Mean by US State - Gender:  Female',geo_scope='usa',
                                 title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})
StateGen_ForMap1_M = go.Figure(data=go.Choropleth(locations = StateGen_ForMapM['state_final'],z = StateGen_ForMapM['mean'],
locationmode = 'USA-states', # set of locations match entries in `locations`
colorscale = 'sunset',colorbar_title = "Total Score Mean"))
StateGen_ForMap1_M.update_layout(dragmode = False,title_text = 'Total Score Mean by US State - Gender:  Male',geo_scope='usa',
                                 title = {'y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'})

cold, cole,colf=st.columns(3)
with cole:
    st.plotly_chart(State_ForMap1)

col23, col24=st.columns(2)
with col23:
    st.plotly_chart(StateGen_ForMap1_F)
with col24:
    st.plotly_chart(StateGen_ForMap1_M)
