# sps-chatbot!

This is a semi-rule-based chatbot meant to address questions specific to my high school. Because of that,
the training and evaluation datasets are small, which led to difficulties in improving accuracy and minimizing loss.
I tested a few different models (dropout layers, embeddings, etc.) before ultimately finding that the one of the most simple
structures was the most effective -- likely because of the limitations of the datasets. It achieved 82% accuracy.
It relies on a classification model to determine the category of the user's question (machine learning),
but the responses are pre-determined based on that category (rule-based). The responses are pretty broad and lengthy for that reason.

I referenced the following articles for figuring out an approach to this project:

https://pykit.org/chatbot-in-python-using-nlp/

https://www.projectpro.io/article/python-chatbot-project-learn-to-build-a-chatbot-from-scratch/429

This was my capstone project I worked on through my high school's Applied Science and Engineering Program (ASEP)!
Since this was a graded project (and reviewed by biology and engineering teachers), my code is heavily, heavily
commented. It was also originally worked on in Google CoLab in order to access the free GPU, but I removed/commented 
out the sections that corresponded to that. Both the training and evaluation datasets are attached.

At the time of creation (fall 2022), I only had a little experience with machine learning, so this was a fun exercise.
