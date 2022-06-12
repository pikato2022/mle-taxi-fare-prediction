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

- Step 3: Edit permission for your Cloud Build services account
```Shell
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
CLOUD_BUILD_SERVICE_ACCOUNT="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:$CLOUD_BUILD_SERVICE_ACCOUNT \
  --role roles/editor
```
- Step 4:
```Shell

```
