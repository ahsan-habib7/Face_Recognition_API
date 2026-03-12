# Face Recognition & ID Validation API

A **Face Recognition and Bangladesh ID Validation API** built using **FastAPI**, **InsightFace (ArcFace)**, and **Tesseract OCR**.

This project verifies whether a **Bangladesh ID card/Passport/License is valid** and checks if the **selfie matches the face on the ID card**.

The entire system runs inside **Docker containers** for easy setup and deployment.

---

# Features

* Bangladesh **ID Card validation using OCR**
* **Face verification** using InsightFace ArcFace
* **Selfie vs ID face comparison**
* **Dockerized backend**
* **FastAPI interactive documentation**

---

# Project Structure

```
InsightFace_BackendAPI/
│
├── __pycache__/                 # Python cache files
│
├── main.py                      # FastAPI main backend application
├── requirements.txt             # Python dependencies
│
├── docker-compose.yml           # Docker compose configuration
├── Dockerfile                   # Docker build instructions
│
├── FaceVerificationService.cs   # .NET service for face verification
├── Result.cshtml                # Web page for displaying results
│
└── README.md                    # Project documentation
```

---

# API Endpoints

## Health Check

```
GET /health
```

Used to verify that the API server is running.

Example response:

```
{
  "status": "API running"
}
```

---

# Validate Bangladesh ID

```
POST /validate-id
```

Uses **Tesseract OCR** to check if the uploaded image contains a **Bangladesh National ID/Passport/License card**.

Request:

Form Data

| Field | Type  |
| ----- | ----- |
| file  | Image |

Response:

```
{
  "valid_id": true,
  "message": "Bangladesh ID detected"
}
```

---

# Face Verification

```
POST /verify-face
```

Compares the **face on the ID card** with a **selfie image** using **ArcFace embeddings**.

Request:

Form Data

| Field        | Type  |
| ------------ | ----- |
| id_image     | Image |
| selfie_image | Image |

Response:

```
{
  "match": true,
  "similarity_score": 0.84
}
```

---

# Running the Project

Go to the project folder and copy the address:
```
C:\Users\Projects\InsightFace_BackendAPI
```

---

# Start the Application

Run the following command:

```
docker-compose up --build
```

This will:

* Build the Docker image
* Install dependencies
* Start the FastAPI server

---

# Stop the Application

```
docker-compose down
```

---

# API Documentation

After starting the container, open the browser:

```
http://localhost:8000/docs
```

FastAPI provides an **interactive Swagger UI** where you can test all endpoints.

---

# Install Dependencies Manually (Optional)

If running without Docker:

```
pip install -r requirements.txt
```

Then start the API:

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

# Technologies Used

* FastAPI
* InsightFace (ArcFace)
* OpenCV
* Tesseract OCR
* Python
* Docker
* Docker Compose

---

# Use Case

This system can be used for:

* **Digital KYC verification**
* **Bank account verification**
* **Online identity validation**
* **Fraud prevention systems**

---

# Author

Developed for a **Face Recognition Full Stack System** using **InsightFace and FastAPI**.
