#lightweight python
FROM python:3.8-slim

# Copy Local code to container image
ENV APP_HOME /app
WORKDIR $APP_HOME
Copy . ./

# Install Dependeicies
RUN pip install -r requirements.txt

# run the flask app on container start
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 repeat_contacts_gunicorn:app

