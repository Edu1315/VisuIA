from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os

load_dotenv()

mongo = PyMongo()


def init_db(app):
    mongo_uri = os.getenv("MONGO_URI")

    app.config["MONGO_URI"] = mongo_uri

    mongo.init_app(app)