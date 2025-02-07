import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import StringIO
from classifier import classify

app = FastAPI()

@app.post("/classify/")
async def classify_logs(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(file.file)

        # Check if necessary columns exist
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        # Apply classification
        df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

        # Convert DataFrame to CSV format in memory (without saving to disk)
        output_buffer = StringIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        # Return CSV as a streaming response
        return StreamingResponse(output_buffer, media_type="text/csv", headers={
            "Content-Disposition": f"attachment; filename=processed_{file.filename}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))