from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Certification Le Wagon (Session Septembre 2024)': 'This is the first app of my new project !'}
