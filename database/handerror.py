from hashlib import sha256

def encrypt(n):
    n = n.encode("utf-8")
    sha = sha256()
    sha.update(n)
    return sha.hexdigest()
