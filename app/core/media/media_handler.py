import base64
import logging

from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)


class MediaHandler:
    def __init__(self):
        pass

    def save_images_to_disk(self, images: list[dict]):
        for image in images:
            image_path = settings.IMAGE_DIR / f"{image['doc_id']}.png"
            image_bytes = base64.b64decode(image["data"])
            try:
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"Image: {image['doc_id']} saved at {image_path}")
            except Exception as e:
                raise HTTPException(
                    status=500, detail=f"Failed to save image: ({image['doc_id']}): {e}"
                ) from e

    def get_image(self, image_id: str) -> str:
        image_path = settings.IMAGE_DIR / f"{image_id}.png"
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {e}"
            ) from e

        image_base64 = base64.b64encode(image_bytes)

        return image_base64

    def save_tables_to_disk(self, tables: list[dict]):
        for tbl in tables:
            table_id = tbl["doc_id"]
            table_md = tbl["data"]
            table_path = settings.TABLE_DIR / f"{table_id}.md"
            try:
                with open(table_path, "w", encoding="utf-8") as f:
                    f.write(table_md)
                logger.info(f"Table: {table_id} saved at {table_path}")
            except Exception as e:
                raise HTTPException(
                    status=500, detail=f"Failed to save table({table_id}): {e}"
                ) from e

    def get_table(self, table_id: str) -> str:
        table_path = settings.TABLE_DIR / f"{table_id}.png"
        try:
            with open(table_path, "rb") as f:
                table = f.read()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save file: {e}"
            ) from e

        return table
