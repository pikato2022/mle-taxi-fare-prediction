# Chicgo Taxi Fare 


# 1. How to setup GCloud Project
- Step 1: Create a new Gcloud Project
- Step 2: Activate the cloud shell:
```Shell
PROJECT_ID=<project-id-name> #should be the name of the project
gcloud config set project $PROJECT_ID

#Enable the cloud services
gcloud services enable \
cloudbuild.googleapis.com \
container.googleapis.com \
cloudresourcemanager.googleapis.com \
iam.googleapis.com \
containerregistry.googleapis.com \
containeranalysis.googleapis.com \
ml.googleapis.com \
dataflow.googleapis.com
```
