import gdown
import os

FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"

OUTPUT = "models/rf_model_new.pkl"

if not os.path.exists(OUTPUT):
    os.makedirs("models", exist_ok=True)
    gdown.download(
        f"https://drive.google.com/file/d/1b_wpi4dSIPETv5wCh_-KB7zcQP5JAUbA/view?usp=sharing",
        OUTPUT,
        quiet=False
    )

   