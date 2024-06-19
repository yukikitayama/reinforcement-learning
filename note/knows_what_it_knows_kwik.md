# Know what it knows (KWIK)

The learning algorithm needs to make only accurate predictions, 
although it can opt out of predictions by saying "I don't know".

In general, any algorithm that guesses a weight vector may never 
find the optimal path.

An algorithm that uses linear algebra to distinguish known from 
unknown costs will either take an optimal route or discover the 
cost of a linearly independent cost vector on each episode.

KWIK allows the learner to opt out of some inputs by returning 
"I don't know".
