#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate No-show Appointments Dataset 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > This dataset provides information about 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. Several characteristics about the patient are included in each row.
# > These are some questions we will try to answer them:
# <li> (1) what is the proportion of females versus males?</li>
# <li> (2) What is the proportion of appointment show-up versus no shows?</li>
# <li>(3) Which gender of patients show up less for their scheduled appointments?</li>
# <li> (4) Does sending an SMS reminder help to reduce the no-shows?</li>
# <li> (5) Is there an assosiation between Scholarship and the no-shows?</li>

# In[80]:


# import the packages we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties

# In[81]:


# Load the data and print out a few lines. 
df = pd.read_csv('Database_No_show_appointments//noshowappointments-kagglev2-may-2016.csv')
df.head()


# In[82]:


# show the dataset dimensions
df.shape


# The data contain 110527 rows and 14 column

# In[83]:


# get the summary of the dataset 
df.describe()


# In[84]:


# show data type of each column and if there is null values
df.info()


# The ScheduledDay and AppointmentDay has incorrect types, so we will convert them to datetime type

# In[85]:


# Convert ScheduledDay to datetime type
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
# Convert AppointmentDay to datetime type
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])


# In[86]:


# show data type of each column again
df.info()


# There is no missing values in the dataset because all the nun-null values are the same of the number of entries (110527) 
# and we can also check for null values using the following code

# In[87]:


# check for the Nan values
df.isnull().sum()


# In[88]:


# check for duplicate records
df.duplicated().sum()


# There is no duplicated records

# 
# 
# ### Data Cleaning 

# The summury of the data shows that the minimum value of the age is (-1), so we will investigate the age column 

# In[89]:


# explore the age value counts
df['Age'].value_counts()


# There is 1 row which must be treated. We will replace this age (-1) with the age's mean. 

# In[90]:


# replace the row containing (-1) in age with the age mean
df['Age'].replace({-1: df['Age'].mean()}, inplace=True)


# There are some mistakes in columns' labels which have to be treated

# In[91]:


# renaming typo mistakes in columns' labels
df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'No-show': 'No_Show'}, inplace=True)


# In[92]:


# show the first 5 rows of the data and its labels
df.head()


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > After cleaning the dataset, we will try to answer our questions.
# 
# ### Research Question 1 ( what is the proportion of females versus males?)

# In[93]:


# count the number of males and femals and then calculate its proportion
print(df.Gender.value_counts())
print(round(df.Gender.value_counts()/len(df)*100))


# There are 71840 female which represents 65% of the dataset and 38687 male which represents 35% of the dataset.
# The following bar chart shows the female versus male value counts.

# In[94]:


# ploting a bar chart for ender
df.Gender.value_counts().plot(kind="bar")
plt.title("The distribution of the dataset according to gender");


# The bar chart show that the females are approximately twice more than males.

# ### Research Question 2 ( What is the proportion of appointment show-up versus no shows?)

# In[95]:


# count the number patients who show and did not show then calculate its proportion
print(df.No_Show.value_counts())
print(round(df.No_Show.value_counts()/len(df)*100))


# There are 88208 patients showed up to their appointment which represents 80% of the dataset
# and 22319 patients didn't show up which represents 20% of the dataset.
# The following bar chart shows the No_Show versus Show value counts.

# In[96]:


df.No_Show.value_counts().plot(kind="pie")
plt.title("The distribution of the dataset according to no show appointment");
plt.legend();


# The pie chart shows that the patients showed up to their appointment are approximately 4 times more than patients didn't show up.

# ### Research Question 3 ( Which gender of patients show up less for their scheduled appointments?)

# We first explore how many females and males who showed up and then calculate the ratio of showing up for each gender

# In[97]:


# grouping the patients who showed up or not according to gender
gender_no_show = df.groupby('Gender')['No_Show'].value_counts()
gender_no_show


# In[98]:


# calculate the ratio of females who showed up from the whole females
F_ratio = round(gender_no_show[0]/(gender_no_show[0]+gender_no_show[1])*100)
F_ratio


# In[99]:


# calculate the ratio of males who showed up from the whole males
M_ratio = round(gender_no_show[2]/(gender_no_show[2]+gender_no_show[3])*100)
M_ratio


# The ratio of females are approximately the same of the ratio of males who showed up.
# we can show that by plotting the following bar chart.

# In[100]:


# define a function that help us explore the relation between No show appointment and any other variable by plotting them
def value_counts_plot(df, x_var):
    # plot
    df.groupby([x_var])['No_Show'].value_counts(normalize=True).mul(100).unstack('No_Show').plot(kind='bar');
    # The title of the plot
    plt.title(f"Patients who show up or not according to {x_var}".title());
    # X axis label
    plt.xlabel(x_var.title());
    # y axis label
    plt.ylabel("Percentage".title());


# In[101]:


# plot a bar chart for patients who show up or not according to gender
value_counts_plot(df, 'Gender')


# The bar chart show that the proportion of females and males who showed up and who did not show up are approximately the same.

# ### Research Question 4 ( Does sending an SMS reminder help to reduce the no-shows?)

# We group the patients who showed up or not according to the SMS reminder.

# In[102]:


# grouping the patients who showed up or not according to SMS reminder
sms_no_show = df.groupby('SMS_received')['No_Show'].value_counts()
print(sms_no_show)


# In[103]:


# plot a bar chart to show the relation between showing up and recieving an SMS
value_counts_plot(df, 'SMS_received')


# The percent of patients who did not show up and recievied an SMS is more than the percent of patients who showed up and recievied an SMS. This means that Sending an SMS reminder does not help to reduce the no-shows appointments.

# ### Research Question 5 ( Is there an assosiation between Scholarship and the no-shows?)

# We group the patients who showed up or not according to Scholarship.

# In[104]:


# grouping the patients who showed up or not according to Scholarship
Scholarship_no_show = df.groupby('Scholarship')['No_Show'].value_counts()
print(Scholarship_no_show)


# In[105]:


# plot a bar chart to show the relation between showing up and Scholarship
value_counts_plot(df, 'Scholarship')


# The percent of patients who did not show up and have Scholarship is more than the percent of patients who showed up and have Scholarship. This means that there is no assosiation between having a scholarship and the no-shows appointments.

# <a id='conclusions'></a>
# ## Conclusions
# 
# > We load the packages we need and read the dadaset of No-show Appointments Dataset. We explore the dataset dimentions and describtion. We clean the dataset and start to find answers for our fifth questions. These answers are: There are 71840 female which represents 65% of the dataset and 38687 male which represents 35% of the dataset. There are 88208 patients showed up to their appointment which represents 80% of the dataset and 22319 patients didn't show up which represents 20% of the dataset. The ratio of females are approximately the same of the ratio of males who showed up. Sending an SMS reminder does not help to reduce the no-shows appointments. There is no assosiation between having a scholarship and the no-shows appointments.
# 
# <a id='Limitations'></a>
# >Limitations:
# the data should contain more effective variables such as the distance between the patient and the place of the appointment, the cost to be paid by the patient would pay, patient's income, and the number of the patient's family members.

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

