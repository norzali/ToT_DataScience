# This code tells Python to import the pandas library, 
# and then tells it that when we use the letters pd,
# we want it to refer to that pandas library.

import pandas as pd


# df = tells Python we’re creating a new variable called df, 
# and when you see df, please refer to the following information:
# pd tells Python to look at the pandas library we imported earlier.

# tells Python to use the function .read_csv() to read the file survey_results_public.csv.
df = pd.read_csv('StudentsPerformance.csv')


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

#df['lunch'].value_counts()


#df['education'].value_counts()


#df['gender'].value_counts()

#df['reading score'].value_counts()


#However, if we set normalize to True, it will “normalize” the counts by representing them 
#as a percentage of the total number of rows in the pandas series we’ve specified.

df['education'].value_counts(normalize="master's degree")

#Let’s try the same thing on another interesting Yes/No question: 
#    “Do you believe that you need to be a manager to make more money?” 
#Many Silicon Valley companies claim that management isn’t the only path to financial success, but are developers buying it?

#df['MgrMoney'].value_counts(normalize=True)

#Once we’ve run that, all we need to do is add a little snippet to the end of our code:
#    .plot(kind='bar'). 
#This tells Python to take whatever we’ve just given it and plot the results in a bar graph.
#(We could replace 'bar' with 'pie' to get a pie chart instead, if we wanted).

#df['reading score'].value_counts().plot(kind="bar")

#Specifically, let’s add two:

#An argument called reading score that defines the size of the chart in the form of a width and height in inches (i.e. (15,7)
#An argument called color that defines the color of the bars.

# #61D199, Dataquest’s green color


#df['reading score'].value_counts().plot(kind="bar", figsize=(15,7), color="#61d199")


said_no = df[df['race'] == 'group A']
said_no.head(3)
said_no.shape

said_yes = df[df['lunch'] == 'standard']


print(said_no['math score'].mean(),
      said_yes['math score'].mean(),
      said_no['math score'].median(),
      said_yes['math score'].median()
     )

over60 = df[df['math score'] >= 60]
under50 = df[df['math score'] <= 50]


print(over60['gender'].value_counts(normalize="female"))
print(under50['gender'].value_counts(normalize="male"))


print(len(over60))
print(len(under50))

filtered_1 = df[(df['gender'] == 'female') & (df['country'] == 'pakistan')]


print(filtered_1['gender'].value_counts())
print(filtered_1['country'].value_counts())
