from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
import json
from typing import Dict, Any, List, Optional
import aiofiles
import logging
from server.config import VOLUMES_PATH

logger = logging.getLogger(__name__)

app = FastAPI(title="Sheet Environment Backend")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "jupyter"}


@app.get("/files")
async def list_files():
    """List available XLSX files."""
    files = []

    # Check sample files
    sample_dir = Path(VOLUMES_PATH)
    if sample_dir.exists():
        for file in sample_dir.glob("*.xlsx"):
            files.append(
                {
                    "name": file.name,
                    "path": str(file),
                    "size": file.stat().st_size,
                    "location": "sample_files",
                }
            )

    # Check notebooks directory
    notebooks_dir = Path("/app/notebooks")
    if notebooks_dir.exists():
        for file in notebooks_dir.glob("*.xlsx"):
            files.append(
                {
                    "name": file.name,
                    "path": str(file),
                    "size": file.stat().st_size,
                    "location": "notebooks",
                }
            )

    return {"files": files}


@app.get("/files/{file_name}/info")
async def file_info(file_name: str):
    """Get information about a specific XLSX file."""
    # Try to find the file
    for directory in [VOLUMES_PATH, "/app/notebooks"]:
        file_path = Path(directory) / file_name
        if file_path.exists() and file_path.suffix == ".xlsx":
            try:
                # Get basic file info
                stat = file_path.stat()
                info = {
                    "name": file_name,
                    "path": str(file_path),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                }

                # Get Excel-specific info
                xl_file = pd.ExcelFile(file_path)
                info["sheets"] = xl_file.sheet_names
                info["sheet_count"] = len(xl_file.sheet_names)

                # Get shape info for each sheet
                sheet_info = {}
                for sheet_name in xl_file.sheet_names:
                    try:
                        df = pd.read_excel(
                            file_path, sheet_name=sheet_name, nrows=0
                        )  # Just get columns
                        sheet_info[sheet_name] = {
                            "columns": list(df.columns),
                            "column_count": len(df.columns),
                        }
                    except Exception as e:
                        sheet_info[sheet_name] = {"error": str(e)}

                info["sheet_details"] = sheet_info
                return info

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    raise HTTPException(status_code=404, detail=f"File {file_name} not found")


@app.get("/files/{file_name}/preview")
async def file_preview(file_name: str, sheet_name: Optional[str] = None, max_rows: int = 10):
    """Get a preview of an XLSX file."""
    # Try to find the file
    for directory in [VOLUMES_PATH, "/app/notebooks"]:
        file_path = Path(directory) / file_name
        if file_path.exists() and file_path.suffix == ".xlsx":
            try:
                xl_file = pd.ExcelFile(file_path)

                # Determine which sheets to preview
                sheets_to_read = (
                    [sheet_name]
                    if sheet_name and sheet_name in xl_file.sheet_names
                    else xl_file.sheet_names
                )

                preview_data = {}
                for sheet in sheets_to_read:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet, nrows=max_rows)
                        preview_data[sheet] = {
                            "shape": df.shape,
                            "columns": list(df.columns),
                            "data": df.to_dict(orient="records"),
                            "dtypes": df.dtypes.astype(str).to_dict(),
                        }
                    except Exception as e:
                        preview_data[sheet] = {"error": str(e)}

                return {"file": file_name, "sheets": preview_data, "max_rows": max_rows}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    raise HTTPException(status_code=404, detail=f"File {file_name} not found")


@app.post("/workspace/save")
async def save_to_workspace(data: Dict[str, Any]):
    """Save data to workspace."""
    try:
        file_name = data.get("file_name")
        content = data.get("content")

        if not file_name or not content:
            raise HTTPException(status_code=400, detail="file_name and content required")

        workspace_dir = Path("/app/notebooks")
        workspace_dir.mkdir(exist_ok=True)

        file_path = workspace_dir / file_name

        # Save based on file extension
        if file_path.suffix == ".json":
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(content, indent=2))
        elif file_path.suffix == ".txt":
            async with aiofiles.open(file_path, "w") as f:
                await f.write(str(content))
        else:
            # Default to JSON
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(content, indent=2))

        return {"status": "saved", "path": str(file_path)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


@app.get("/workspace")
async def list_workspace():
    """List files in the workspace."""
    workspace_dir = Path("/app/notebooks")
    files = []

    if workspace_dir.exists():
        for file in workspace_dir.iterdir():
            if file.is_file():
                files.append(
                    {
                        "name": file.name,
                        "path": str(file),
                        "size": file.stat().st_size,
                        "extension": file.suffix,
                    }
                )

    return {"workspace": "/app/notebooks", "files": files}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
