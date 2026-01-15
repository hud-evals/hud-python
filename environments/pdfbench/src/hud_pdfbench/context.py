"""PDF environment context for state management."""

import json
import logging
import os
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def bbox_key(page_num: int, rect: fitz.Rect) -> str:
    """Create a stable key from page number and bounding box coordinates."""
    return f"{page_num},{round(rect.x0)},{round(rect.y0)},{round(rect.x1)},{round(rect.y1)}"


class PDFContext:
    """Holds the current PDF state."""

    def __init__(self):
        self.doc: fitz.Document | None = None
        self.pdf_path: str | None = None
        self.output_path: str | None = None
        self.solution_path: str | None = None
        self.original_pdf_path: str | None = None

    def reset(self):
        """Reset the context state."""
        if self.doc:
            self.doc.close()
        self.doc = None
        self.pdf_path = None
        self.output_path = None
        self.solution_path = None
        self.original_pdf_path = None

    def load_pdf(
        self,
        pdf_path: str,
        output_path: str | None = None,
        solution_path: str | None = None,
    ) -> dict[str, Any]:
        """Load a blank PDF form for filling."""
        self.reset()

        if not os.path.exists(pdf_path):
            return {"error": f"PDF not found: {pdf_path}"}

        try:
            self.doc = fitz.open(pdf_path)
            self.pdf_path = pdf_path
            self.original_pdf_path = pdf_path
            self.output_path = output_path or "/tmp/filled.pdf"
            self.solution_path = solution_path

            # Count fields
            field_count = 0
            fields = []
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                for widget in page.widgets():
                    field_count += 1
                    fields.append({
                        "name": widget.field_name,
                        "type": widget.field_type_string,
                        "page": page_num,
                        "bbox": bbox_key(page_num, widget.rect),
                    })

            logger.info(f"Loaded PDF with {field_count} form fields: {pdf_path}")

            return {
                "success": True,
                "message": f"Loaded PDF with {field_count} form fields",
                "pdf_path": pdf_path,
                "output_path": self.output_path,
                "field_count": field_count,
                "fields": fields[:20],
                "total_fields": field_count,
            }
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            return {"error": str(e)}

    def list_fields(self, page: int | None = None) -> dict[str, Any]:
        """List all form fields."""
        if not self.doc:
            return {"error": "No PDF loaded. Call setup first."}

        fields = []
        for page_num in range(len(self.doc)):
            if page is not None and page_num != page:
                continue
            page_obj = self.doc[page_num]
            for widget in page_obj.widgets():
                fields.append({
                    "name": widget.field_name,
                    "type": widget.field_type_string,
                    "value": widget.field_value,
                    "page": page_num,
                    "bbox": bbox_key(page_num, widget.rect),
                })

        return {"field_count": len(fields), "fields": fields}

    def fill_field(
        self,
        field_name: str | None = None,
        bbox: str | None = None,
        value: str = "",
    ) -> dict[str, Any]:
        """Fill a form field."""
        if not self.doc:
            return {"error": "No PDF loaded. Call setup first."}

        if not field_name and not bbox:
            return {"error": "Must provide either field_name or bbox"}

        target_widget = None
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            for widget in page.widgets():
                if field_name and widget.field_name == field_name:
                    target_widget = widget
                    break
                if bbox and bbox_key(page_num, widget.rect) == bbox:
                    target_widget = widget
                    break
            if target_widget:
                break

        if not target_widget:
            return {"error": f"Field not found: {field_name or bbox}"}

        try:
            if target_widget.field_type == fitz.PDF_WIDGET_TYPE_CHECKBOX:
                target_widget.field_value = value.lower() in ['yes', 'true', '1', 'on', 'checked']
            else:
                target_widget.field_value = value
            target_widget.update()

            return {
                "success": True,
                "field_name": target_widget.field_name,
                "value": value,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_field(self, field_name: str | None = None, bbox: str | None = None) -> dict[str, Any]:
        """Get a field's current value."""
        if not self.doc:
            return {"error": "No PDF loaded. Call setup first."}

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            for widget in page.widgets():
                if field_name and widget.field_name == field_name:
                    return {
                        "field_name": widget.field_name,
                        "value": widget.field_value,
                        "type": widget.field_type_string,
                        "bbox": bbox_key(page_num, widget.rect),
                    }
                if bbox and bbox_key(page_num, widget.rect) == bbox:
                    return {
                        "field_name": widget.field_name,
                        "value": widget.field_value,
                        "type": widget.field_type_string,
                        "bbox": bbox,
                    }

        return {"error": f"Field not found: {field_name or bbox}"}

    def save_pdf(self, output_path: str | None = None) -> dict[str, Any]:
        """Save the filled PDF."""
        if not self.doc:
            return {"error": "No PDF loaded. Call setup first."}

        save_path = output_path or self.output_path or "/tmp/filled.pdf"

        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.doc.save(save_path)
            self.output_path = save_path
            return {"success": True, "path": save_path}
        except Exception as e:
            return {"error": str(e)}

    def get_field_values_by_bbox(self, filepath: str | None = None) -> dict[str, dict[str, Any]]:
        """Extract field values keyed by bbox."""
        doc = fitz.open(filepath) if filepath else self.doc
        if not doc:
            return {}

        field_values = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            for widget in page.widgets():
                key = bbox_key(page_num, widget.rect)
                field_values[key] = {
                    "value": widget.field_value,
                    "field_name": widget.field_name,
                    "field_type": widget.field_type,
                }

        if filepath:
            doc.close()
        return field_values


# Global context instance
pdf_context = PDFContext()
