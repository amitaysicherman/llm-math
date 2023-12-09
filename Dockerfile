# gcloud builds submit --tag gcr.io/llm-math-406622/llm-math-vis  --project=llm-math-406622 | bar
# gcloud run deploy --image gcr.io/llm-math-406622/llm-math-vis --platform managed  --project=llm-math-406622 --allow-unauthenticated

FROM python:3.11

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

EXPOSE 8050
CMD python app.py