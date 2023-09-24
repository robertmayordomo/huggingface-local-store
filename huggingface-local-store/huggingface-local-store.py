import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
from azure.storage.blob import BlobServiceClient


class AzureModelService:
    # Define constants
    container_name = ""
    container_folder = ""
    local_model_path = ""
    hugging_face_model = ""
    azure_storage_connection_string = ""
    base_directory = ""
    model_directory = ""

    def __init__(
        self,
        azure_storage_connection_string=None,
        container_name=None,
        container_folder=None,
        local_model_path=None,
        hugging_face_model=None,
    ):
        if container_name is None or not container_name.strip():
            raise ValueError("container_name cannot be None or empty.")
        if container_folder is None or not container_folder.strip():
            raise ValueError("container_folder cannot be None or empty.")
        if local_model_path is None or not local_model_path.strip():
            raise ValueError("local_model_path cannot be None or empty.")
        if hugging_face_model is None or not local_model_path.strip():
            raise ValueError("hugging_face_model cannot be None or empty.")
        if (azure_storage_connection_string is None or not azure_storage_connection_string.strip()):
            raise ValueError("azure_storage_connection_string cannot be None or empty.")

        self.container_name = container_name
        self.container_folder = container_folder
        self.local_model_path = local_model_path
        self.hugging_face_model = hugging_face_model
        self.azure_storage_connection_string = azure_storage_connection_string
        self.base_directory = os.getcwd()
        self.model_directory = os.path.join(self.base_directory, local_model_path)

    def save_to_local(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hugging_face_model)
        model = AutoModelForTokenClassification.from_pretrained(self.hugging_face_model)

        model.save_pretrained(self.local_model_path)
        tokenizer.save_pretrained(self.local_model_path)

    def save_to_azure(self):
        client = self.get_client(self.azure_storage_connection_string)

        self.save_local_safe(client, "/config.json")
        self.save_local_safe(client, "/special_tokens_map.json")
        self.save_local_safe(client, "/tokenizer_config.json")
        self.save_local_safe(client, "/tokenizer.json")
        self.save_local_safe(client, "/vocab.txt")
        self.save_local_safe(client, "/pytorch_model.bin")

    def save_local_safe(self, client, file_path):
        try:
            with open(self.container_name + file_path, "rb") as file:
                client.upload_blob(self.container_folder + file_path, file)

        except Exception as e:
            # Code to handle the exception
            print(f"An exception of type {type(e).__name__} occurred: {e}")
        finally:
            print(f"Finished saving {file_path}")

    def get_models_from_azure(self):
        self.get_from_azure()

        tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
        model = AutoModelForTokenClassification.from_pretrained(self.model_directory)
        return model, tokenizer

    def get_from_azure(self):
        if not os.path.exists(self.local_model_path):
            os.makedirs(self.local_model_path)

        client = self.get_client(self.azure_storage_connection_string)

        # Define a dictionary to map file names to blob names
        file_blob_mapping = {
            "config.json": "config.json",
            "special_tokens_map.json": "special_tokens_map.json",
            "tokenizer_config.json": "tokenizer_config.json",
            "tokenizer.json": "tokenizer.json",
            "vocab.txt": "vocab.txt",
            "pytorch_model.bin": "pytorch_model.bin",
        }

        # Iterate over the file_blob_mapping and download each blob
        for file_name, blob_name in file_blob_mapping.items():
            # Build the file path
            file_path = os.path.join(
                self.base_directory, self.local_model_path, file_name
            )
            if os.path.exists(file_path):
                continue

            # Download the blob
            blob_client = client.get_blob_client(
                os.path.join(self.container_folder, blob_name)
            )
            data = blob_client.download_blob()

            # Open the file and write the blob's content
            with open(file_path, "wb") as file:
                file.write(data.readall())

    def get_client(self, connection_string):
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        return blob_service_client.get_container_client(self.container_name)

    def get_models(self):
        tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        model = AutoModelForTokenClassification.from_pretrained(self.local_model_path)
        return model, tokenizer
