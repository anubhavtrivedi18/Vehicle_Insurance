from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import (
    VehicleData,
    VehicleDataClassifier
)
from src.pipline.training_pipeline import TrainPipeline


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Vehicle Insurance Prediction API",
    description="Machine Learning API for Vehicle Insurance Prediction",
    version="1.0.0",
)


# ============================================================
# STATIC FILES
# ============================================================

STATIC_DIR = BASE_DIR / "static"

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ============================================================
# JINJA2 TEMPLATES
# ============================================================

TEMPLATE_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# ============================================================
# CORS
# ============================================================

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATA FORM
# ============================================================

class DataForm:
    """
    Handles vehicle insurance form data.
    """

    def __init__(self, request: Request):

        self.request = request

        self.Gender: Optional[int] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[float] = None
        self.Previously_Insured: Optional[int] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[float] = None
        self.Vintage: Optional[int] = None
        self.Vehicle_Age_lt_1_Year: Optional[int] = None
        self.Vehicle_Age_gt_2_Years: Optional[int] = None
        self.Vehicle_Damage_Yes: Optional[int] = None

    async def get_vehicle_data(self):
        """
        Extract and convert form data.
        """

        form = await self.request.form()

        self.Gender = int(form.get("Gender"))
        self.Age = int(form.get("Age"))
        self.Driving_License = int(
            form.get("Driving_License")
        )
        self.Region_Code = float(
            form.get("Region_Code")
        )
        self.Previously_Insured = int(
            form.get("Previously_Insured")
        )
        self.Annual_Premium = float(
            form.get("Annual_Premium")
        )
        self.Policy_Sales_Channel = float(
            form.get("Policy_Sales_Channel")
        )
        self.Vintage = int(
            form.get("Vintage")
        )
        self.Vehicle_Age_lt_1_Year = int(
            form.get("Vehicle_Age_lt_1_Year")
        )
        self.Vehicle_Age_gt_2_Years = int(
            form.get("Vehicle_Age_gt_2_Years")
        )
        self.Vehicle_Damage_Yes = int(
            form.get("Vehicle_Damage_Yes")
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", tags=["Home"])
async def index(request: Request):
    """
    Render the vehicle insurance prediction form.
    """

    return templates.TemplateResponse(
        request=request,
        name="vehicledata.html",
        context={
            "request": request,
            "context": None
        }
    )


# ============================================================
# TRAINING ROUTE
# ============================================================

@app.get("/train", tags=["Training"])
async def train_route_client():
    """
    Trigger the ML training pipeline.
    """

    try:

        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return {
            "status": True,
            "message": "Training successful!"
        }

    except Exception as e:

        return {
            "status": False,
            "message": "Training failed",
            "error": str(e)
        }


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.post("/", tags=["Prediction"])
async def predict_route_client(request: Request):
    """
    Receive vehicle information and return prediction.
    """

    try:

        # ----------------------------------------------------
        # Get form data
        # ----------------------------------------------------

        form = DataForm(request)

        await form.get_vehicle_data()


        # ----------------------------------------------------
        # Create VehicleData object
        # ----------------------------------------------------

        vehicle_data = VehicleData(

            Gender=form.Gender,

            Age=form.Age,

            Driving_License=form.Driving_License,

            Region_Code=form.Region_Code,

            Previously_Insured=form.Previously_Insured,

            Annual_Premium=form.Annual_Premium,

            Policy_Sales_Channel=form.Policy_Sales_Channel,

            Vintage=form.Vintage,

            Vehicle_Age_lt_1_Year=form.Vehicle_Age_lt_1_Year,

            Vehicle_Age_gt_2_Years=form.Vehicle_Age_gt_2_Years,

            Vehicle_Damage_Yes=form.Vehicle_Damage_Yes
        )


        # ----------------------------------------------------
        # Convert input into DataFrame
        # ----------------------------------------------------

        vehicle_df = (
            vehicle_data
            .get_vehicle_input_data_frame()
        )


        # ----------------------------------------------------
        # Initialize prediction pipeline
        # ----------------------------------------------------

        model_predictor = VehicleDataClassifier()


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = model_predictor.predict(
            dataframe=vehicle_df
        )

        value = prediction[0]


        # ----------------------------------------------------
        # Convert prediction to readable result
        # ----------------------------------------------------

        status = (
            "Response-Yes"
            if value == 1
            else "Response-No"
        )


        # ----------------------------------------------------
        # Render HTML page with result
        # ----------------------------------------------------

        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={
                "request": request,
                "context": status
            }
        )


    except Exception as e:

        return {
            "status": False,
            "message": "Prediction failed",
            "error": str(e)
        }


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )