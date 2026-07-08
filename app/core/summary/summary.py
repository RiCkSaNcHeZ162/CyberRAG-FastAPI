import asyncio
import logging

from app.api.schemas import PdfData
from app.core.llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


class PDFSummary:
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager

    async def summarize_pdf_pages(self, processed_pages: list[PdfData]) -> None:
        text_processed = 0
        for index, page in enumerate(processed_pages):
            if page.type == "image":
                if index != 0 or text_processed != 0:
                    logger.info("waiting for 5 seconds to call the function")
                    await asyncio.sleep(5)
                await self.summarize_images(page)
            elif page.type == "table":
                if index != 0 or text_processed != 0:
                    logger.info("waiting for 5 seconds to call the function")
                    await asyncio.sleep(5)
                await self.summarize_tables(page)
            else:
                text_processed += 1
                continue  # skip non-image/table pages

    async def summarize_images(self, page: PdfData) -> None:
        images = page.data.get("images", [])
        for image in images:
            doc_id = image["img_id"]
            # prompt to summarize image
            prompt = """doc_id: {doc_id}

        Describe the image across these four dimensions:

        1. VISUAL STRUCTURE
        - Type (bar/pie/line/scatter, table, diagram, flowchart, photo, illustration)
        - Layout: orientation, number of sections/bars/slices/nodes
        - Colors used and what they represent
        - Legend, title, axes, labels, annotations present?

        2. CONTENT AND DATA
        - Subject matter
        - All visible text: titles, axis labels, data labels, legends, captions
        - All visible numbers and what they represent
        - Units of measurement

        3. INSIGHTS AND ANALYSIS
        - Key takeaway or main message
        - Comparisons being made
        - Highest, lowest, or most significant item
        - Trends, patterns, anomalies, outliers

        4. QUESTIONS THIS IMAGE CAN ANSWER
        - 4-5 specific natural-language questions this image directly answers

        5. DOC ID
        - Include doc_id ({doc_id}) for traceability

        Be specific and exhaustive — detail improves retrieval accuracy.
        """
            system_prompt = """You describe images for semantic search retrieval. Descriptions become vector embeddings, so cover all details a user might search for."""
            try:
                response = await self.llm_manager.generate(
                    prompt=prompt.format(doc_id=doc_id),
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=20,
                    images=[image],
                )

            except Exception as e:
                logger.warning(f"Action decision failed: {e}. Defaulting to RETRIEVE.")
                continue

            summary = response.strip().lower().replace('"', "").replace("'", "")
            image["text"] = summary
            del image["data"]  # do not store in vectorstore

    async def summarize_tables(self, page: PdfData) -> None:
        tables = page.data.get("tables", [])
        for table in tables:
            table_markdown = table["data"]
            doc_id = table["tbl_id"]
            # prompt to summarize table
            prompt = """ doc_id: {doc_id}
                You are indexing a table extracted from a document for a    
                    semantic search system. Your summary will be converted into
                    a vector embedding, so it must be detailed enough to match
                    any reasonable question a user might ask about this table.

                    Here is the table in markdown format:

                    {table_markdown}

                    Describe the table across these four dimensions:

                    1. STRUCTURE
                        - How many rows and columns does it have?
                        - What are the column headers and what do they represent?
                        - Are there any merged cells, sub-headers, or grouped rows?
                        - Unit of measurement for each column if present?

                    2. CONTENT AND DATA
                        - What is this table about? What domain or topic?
                        - List every row with its key values explicitly
                        so numbers are searchable
                        - Mention the range of values for each numeric column
                        (min, max)

                    3. INSIGHTS AND ANALYSIS
                        - Which row has the highest / lowest value for each
                        numeric column?
                        - Are there any visible trends across rows?
                        - Any outliers or anomalies?
                        - Total or average of numeric columns if meaningful?
                        - Key takeaway from this table?

                    4. QUESTIONS THIS TABLE CAN ANSWER
                        - Write 5-6 specific natural language questions that
                        this table directly answers
                        - Include questions about specific values, comparisons,
                        rankings, and trends
                        - Phrase them exactly as a user would ask them
                    5. DOC ID
                        - Include the doc_id in your summary for traceability
                    """
            system_prompt = """You are indexing a table extracted from a document for a
                    semantic search system. Your summary will be converted into
                    a vector embedding, so it must be detailed enough to match
                    any reasonable question a user might ask about this table."""
            try:
                response = await self.llm_manager.generate(
                    prompt=prompt.format(table_markdown=table_markdown, doc_id=doc_id),
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=200,
                )
            except Exception as e:
                logger.warning(f"Action decision failed: {e}. Defaulting to RETRIEVE.")

            summary = response.strip().lower().replace('"', "").replace("'", "")
            table["text"] = summary
            del table["data"]  # do not save in vectorstore
