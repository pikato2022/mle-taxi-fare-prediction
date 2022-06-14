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
- Step 4: Now, create a custom service account to give CAIP training job access to AI Platform Vizier service for pipeline hyperparameter tuning.
```Shell
SERVICE_ACCOUNT_ID=tfx-tuner-caip-service-account
gcloud iam service-accounts create $SERVICE_ACCOUNT_ID  \
  --description="A custom service account for CAIP training job to access AI Platform Vizier service for pipeline hyperparameter tuning." \
  --display-name="TFX Tuner CAIP Vizier"
```

- Step 5: Grant your AI Platform service account additional access permissions to the AI Platform Vizier service for pipeline hyperparameter tuning.
```Shell
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
CAIP_SERVICE_ACCOUNT="service-${PROJECT_NUMBER}@cloud-ml.google.com.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:$CAIP_SERVICE_ACCOUNT \
  --role=roles/storage.objectAdmin
````
```Shell
 gcloud projects add-iam-policy-binding $PROJECT_ID \
 --member serviceAccount:$CAIP_SERVICE_ACCOUNT \
 --role=roles/ml.admin
```

- Step 6: Grant service account access to Storage admin role.
```Shell
SERVICE_ACCOUNT_ID=tfx-tuner-caip-service-account
gcloud projects add-iam-policy-binding $PROJECT_ID \
--member=serviceAccount:${SERVICE_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com \
--role=roles/storage.objectAdmin
```

- Step 7: Grant service acount access to AI Platform Vizier role.
```Shell
gcloud projects add-iam-policy-binding $PROJECT_ID \
--member=serviceAccount:${SERVICE_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com \
--role=roles/ml.admin
```

- Step 8: Grant your project's AI Platform Google-managed service account the Service Account Admin role for your AI Platform service account.
```Shell
gcloud iam service-accounts add-iam-policy-binding \
 --role=roles/iam.serviceAccountAdmin \
 --member=serviceAccount:service-${PROJECT_NUMBER}@cloud-ml.google.com.iam.gserviceaccount.com \
${SERVICE_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com
```
