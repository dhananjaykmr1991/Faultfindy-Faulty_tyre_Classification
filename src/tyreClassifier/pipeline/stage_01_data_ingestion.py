from tyreClassifier.components.data_ingestion import *
from tyreClassifier.logging import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = DataIngestionConfig()
        data_ingestion = DataIngestion(config=config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e