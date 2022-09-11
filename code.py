import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("scores.csv")
data = pd.read_csv("escape_velocity.csv")
score = data["Score"].tolist()
accepted = data["Accepted"].tolist()
velocity = data["Velocity"].tolist()
escaped = data["Escaped"].tolist()
graph = px.scatter(x = score, y = accepted)
graph = px.scatter(x = velocity, y = escaped)

graph.show()

#  (1/(1 + e^-x)) 

x = np.reshape(score , (len(score) , 1) )
y = np.reshape(accepted , (len(accepted) , 1) )
x = np.reshape(velocity , (len(velocity) , 1) )
y = np.reshape(escaped , (len(escaped) , 1) )

lr = LogisticRegression()
lr.fit(x , y)

plt.figure()
plt.scatter(x.ravel(), y, color='black', zorder=20)

def model(x):
    return 1 / (1 + np.exp(-x))

X_test = np.linspace(0,100,200)

chances = model(X_test * lr.coef_ + lr.intercept_).ravel()



plt.plot(X_test, chances, color='red', linewidth=3)
plt.axhline(y=0, color='k', linestyle='-')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=0.5, color='b', linestyle='--')

plt.axvline(x=X_test[165], color='b', linestyle='--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(75, 85)
plt.show()


