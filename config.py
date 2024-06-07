SAVE_ROOT = "Record"

import os
path = os.getcwd()
print(path)
class settings:
    SECRET_KEY: str = "ATTT"
    JWT_ALGORITHM: str = "HS256"
    HOST: str = "http://127.0.0.1:5000/"