Phishing-Website-Classification 

We have a dataset that contains meta data about 5000 logitimate and 5000 phishing sites. Phishing is a type of semantic attack often used to steal user sensitive information including login credentials and credit card numbers by masquerading as a trusted entity, enticing a victim into clicking on a link or opening an attachment in an email or instant message. This is a large dataset with many meta data information.

Details of the project - 

The project is divided into 3 parts 
1. Building a machine learning model to predict if a site is a phishing site or not
   So we build different machine learning model to find out which one will be most suitable for the data points. After finding out RandomForestClassifier, the appropriate one we use RandomForestClassifier model and
   train that model using phishing site datapoints. There is abundant information meta data of sites which helps in recognizing the phishing sites.

2. Build a web app on Django to mount the prediction model
   In this part of the project we build a web application using Django framework and mount the model we built in the previous part.

3. Deploy the Django project on AWS
   In this part of the project we deploy the Django application on AWS.
