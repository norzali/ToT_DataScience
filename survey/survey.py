# This code tells Python to import the pandas library, 
# and then tells it that when we use the letters pd,
# we want it to refer to that pandas library.

import pandas as pd


# df = tells Python we’re creating a new variable called df, 
# and when you see df, please refer to the following information:
# pd tells Python to look at the pandas library we imported earlier.

# tells Python to use the function .read_csv() to read the file survey_results_public.csv.
df = pd.read_csv('Sustainable_Product_Servi_eSystem_Elements.csv')


#We do need to tell .head() what DataFrame to look at, 
#though, so we’ll use the syntax df.head(). 
#The df tells Python we want to look at that DataFrame we just made with our CSV data, the .
#tells Python that we’re going to do something to that DataFrame, and then the head()
#tells Python what we want to do: display the first five rows.

df.head()

# .shape to give us the size of our data set.

df.shape

#Analyzing Multiple Choice Survey Questions

#The value_counts() function looks at a single column of data at a time and counts how many instances of each unique entry that column contains.

df['Occupation'].value_counts()


df['Education'].value_counts()


df['Gender'].value_counts()

#However, if we set normalize to True, it will “normalize” the counts by representing them 
#as a percentage of the total number of rows in the pandas series we’ve specified.

#df['Occupation'].value_counts(normalize="Studying")



#Let’s try the same thing on another interesting Yes/No question: 
#    “Do you believe that you need to be a manager to make more money?” 
#Many Silicon Valley companies claim that management isn’t the only path to financial success, but are developers buying it?

#df['MgrMoney'].value_counts(normalize=True)

#Once we’ve run that, all we need to do is add a little snippet to the end of our code:
#    .plot(kind='bar'). 
#This tells Python to take whatever we’ve just given it and plot the results in a bar graph.
#(We could replace 'bar' with 'pie' to get a pie chart instead, if we wanted).

#df['SocialMedia'].value_counts().plot(kind="bar")

#Specifically, let’s add two:

#An argument called figsize that defines the size of the chart in the form of a width and height in inches (i.e. (15,7)
#An argument called color that defines the color of the bars.

# #61D199, Dataquest’s green color


#df['SocialMedia'].value_counts().plot(kind="bar", figsize=(15,7), color="#61d199")


#said_no = df[df['BetterLife'] == 'No']
#said_no.head(3)
#said_no.shape
#df['BetterLife'].value_counts()

#said_yes = df[df['BetterLife'] == 'Yes']


#print(said_no['Age'].mean(),
 #     said_yes['Age'].mean(),
  #    said_no['Age'].median(),
   #   said_yes['Age'].median()
    # )

#over50 = df[df['Age'] >= 50]
#under25 = df[df['Age'] <= 25]


#print(over50['BetterLife'].value_counts(normalize=True))
#print(under25['BetterLife'].value_counts(normalize=True))


#print(len(over50))
#print(len(under25))

#filtered_1 = df[(df['BetterLife'] == 'Yes') & (df['Country'] == 'Malaysia')]


#print(filtered_1['BetterLife'].value_counts())
#print(filtered_1['Country'].value_counts())

#filtered = df[(df['BetterLife'] == 'Yes') & (df['Age'] >= 50) & (df['Country'] == 'India') &~ (df['Hobbyist'] == "Yes") &~ (df['OpenSourcer'] == "Never")]

#df["LanguageWorkedWith"].head()


#python_bool = df["LanguageWorkedWith"].str.contains('Python')
#python_bool.value_counts(normalize=True)


#lang_lists = df["LanguageWorkedWith"].str.split(';', expand=True)
#lang_lists.head()

#lang_df.stack().value_counts().plot(kind='bar', figsize=(15,7), color="#61d199")
