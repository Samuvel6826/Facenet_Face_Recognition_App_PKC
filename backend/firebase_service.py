import os
import pickle
from io import BytesIO
import firebase_admin
from firebase_admin import credentials, storage
from PIL import Image
import torch
import numpy as np
from logger_config import setup_logger

logger = setup_logger()

class FirebaseService:
    def __init__(self, firebase_secret_path, mtcnn, model, device):
        self.mtcnn = mtcnn
        self.model = model
        self.device = device
        self.bucket = self._initialize_firebase(firebase_secret_path)
        
    def _initialize_firebase(self, firebase_secret_path):
        try:
            if not os.path.exists(firebase_secret_path):
                logger.error(f"{firebase_secret_path} not found. Make sure the secret file is correctly set up.")
                raise FileNotFoundError(f"{firebase_secret_path} not found.")
                
            cred = credentials.Certificate(firebase_secret_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'face-recognition-storage.appspot.com'
            })
            bucket = storage.bucket()
            logger.info("Firebase Admin SDK initialized successfully.")
            return bucket
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
            raise

    def force_reload_from_firebase(self):
        """
        Force reload all images from Firebase Storage, process them, and save new embeddings
        without using any cached data.
        """
        logger.info("Starting force reload of embeddings from Firebase Storage...")
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        known_encodings = {}
        
        try:
            # List all person folders in known_people directory
            blobs = self.bucket.list_blobs(prefix='known_people/')
            
            for blob in blobs:
                if blob.name.endswith('/') and blob.name != 'known_people/':
                    person_name = blob.name.split('/')[-2]
                    logger.info(f"Processing images for: {person_name}")
                    person_images = []

                    # Process all images for this person
                    person_blobs = self.bucket.list_blobs(prefix=f'{blob.name}')
                    for person_blob in person_blobs:
                        file_extension = os.path.splitext(person_blob.name)[1].lower()
                        if file_extension in allowed_extensions:
                            try:
                                logger.info(f"  Processing {person_blob.name}")
                                img_bytes = person_blob.download_as_bytes()
                                img = Image.open(BytesIO(img_bytes)).convert("RGB")

                                img_cropped = self.mtcnn(img)
                                if img_cropped is not None and len(img_cropped) > 0:
                                    with torch.no_grad():
                                        embedding = self.model(img_cropped.to(self.device)).detach().cpu().numpy()
                                        embedding = embedding.reshape(1, 512)
                                        person_images.append((embedding, person_blob.name.split("/")[-1]))
                                else:
                                    logger.warning(f"No face found in image: {person_blob.name}")
                            except Exception as e:
                                logger.error(f"Error processing image {person_blob.name}: {e}")
                                continue

                    if person_images:
                        known_encodings[person_name] = person_images
                        logger.info(f"Processed {len(person_images)} images for {person_name}")
                    else:
                        logger.warning(f"No valid images processed for {person_name}")

            # Save the new embeddings both locally and to Firebase
            if known_encodings:
                self.save_embeddings(known_encodings)
                logger.info("Successfully saved new embeddings")
            else:
                logger.warning("No embeddings were generated during force reload")

            return known_encodings

        except Exception as e:
            logger.error(f"Error during force reload: {e}")
            raise

    def load_embeddings(self):
        local_cache_path = 'known_embeddings.pkl'
        known_encodings = {}

        if os.path.exists(local_cache_path):
            with open(local_cache_path, 'rb') as f:
                known_encodings = pickle.load(f)
            logger.info("Loaded embeddings from local cache.")
        else:
            try:
                blob = self.bucket.blob('embeddings/known_embeddings.pkl')
                if blob.exists():
                    logger.info("Local cache not found. Downloading embeddings from Firebase.")
                    img_bytes = blob.download_as_bytes()
                    known_encodings = pickle.loads(img_bytes)
                    self.save_embeddings(known_encodings)
                    logger.info("Downloaded embeddings from Firebase and saved to local cache.")
                else:
                    logger.error("No embeddings found in Firebase.")
            except Exception as e:
                logger.error(f"Error downloading embeddings from Firebase: {e}")

        return known_encodings

    def save_embeddings(self, known_encodings):
        local_cache_path = 'known_embeddings.pkl'
        
        with open(local_cache_path, 'wb') as f:
            pickle.dump(known_encodings, f)

        try:
            blob = self.bucket.blob('embeddings/known_embeddings.pkl')
            blob.upload_from_filename(local_cache_path)
            logger.info("Saved embeddings to Firebase.")
        except Exception as e:
            logger.error(f"Error uploading embeddings to Firebase: {e}")

    def load_known_people_images(self):
        known_encodings = self.load_embeddings()
        if known_encodings:
            # Verify and fix existing embeddings
            fixed_encodings = {}
            needs_update = False
            
            for name, encodings in known_encodings.items():
                fixed_encodings[name] = []
                for embedding, filename in encodings:
                    # Ensure embedding has correct shape
                    if embedding.size == 512:  # Correct number of features
                        fixed_embedding = embedding.reshape(1, 512)
                        fixed_encodings[name].append((fixed_embedding, filename))
                        # Check if reshape was needed
                        if not np.array_equal(embedding, fixed_embedding):
                            needs_update = True
                    else:
                        logger.warning(f"Skipping invalid embedding for {name} - {filename}")
                        needs_update = True
            
            if needs_update:
                self.save_embeddings(fixed_encodings)
                logger.info("Fixed and saved embeddings with correct shapes")
            return fixed_encodings

        # Load new embeddings if none exist
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        try:
            blobs = self.bucket.list_blobs(prefix='known_people/')
            known_encodings = {}
            
            for blob in blobs:
                if blob.name.endswith('/') and blob.name != 'known_people/':
                    person_name = blob.name.split('/')[-2]
                    logger.info(f"Loading images for: {person_name}")
                    person_images = []

                    person_blobs = self.bucket.list_blobs(prefix=f'{blob.name}')
                    for person_blob in person_blobs:
                        file_extension = os.path.splitext(person_blob.name)[1].lower()
                        if file_extension in allowed_extensions:
                            try:
                                logger.info(f"  Processing {person_name} from {person_blob.name}")
                                img_bytes = person_blob.download_as_bytes()
                                img = Image.open(BytesIO(img_bytes)).convert("RGB")

                                img_cropped = self.mtcnn(img)
                                if img_cropped is not None and len(img_cropped) > 0:
                                    with torch.no_grad():
                                        embedding = self.model(img_cropped.to(self.device)).detach().cpu().numpy()
                                        embedding = embedding.reshape(1, 512)
                                        person_images.append((embedding, person_blob.name.split("/")[-1]))
                                else:
                                    logger.warning(f"No face found in image: {person_blob.name}")
                            except Exception as e:
                                logger.error(f"Error processing image {person_blob.name}: {e}")

                    if person_images:
                        known_encodings[person_name] = person_images
                        logger.info(f"Loaded {len(person_images)} images for {person_name}")

            logger.info("Finished loading known people images")
            self.save_embeddings(known_encodings)
            return known_encodings

        except Exception as e:
            logger.error(f"Error loading known people images: {e}")
            return {}