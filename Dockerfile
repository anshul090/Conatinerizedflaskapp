#lightweight python
FROM python:3.8-slim

# Copy Local code to container image
ENV APP_HOME /app
ENV PORT 9096
WORKDIR $APP_HOME
Copy . ./

# Install Dependeicies
RUN pip3 install --upgrade pip 

RUN pip3 --no-cache-dir install -r requirements.txt



# run the flask app when container starts 
CMD python repeat_contacts_gunicorn.py


# The below command starts the app with gunicorn multithreaded server on google cloud 
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 repeat_contacts_gunicorn:app


