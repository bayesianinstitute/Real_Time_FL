from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa

def generate_rsa_keypair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    return private_key, public_key

def encrypt_with_fernet(file_contents, aes_key):
    fernet = Fernet(aes_key)
    encrypted_file = fernet.encrypt(file_contents)
    return encrypted_file

def encrypt_with_rsa(public_key, aes_key):
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_aes_key

def decrypt_with_rsa(private_key, encrypted_aes_key):
    decrypted_aes_key = private_key.decrypt(
        encrypted_aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return decrypted_aes_key

if __name__ == "__main__":
    # Generate RSA key pair
    private_key, public_key = generate_rsa_keypair()

    # Example file content to encrypt
    file_contents = b"Hello, this is a secret message!"

    # Generate an AES key (this should be a securely generated key in practice)
    aes_key = Fernet.generate_key()

    # Encrypt the file content using Fernet encryption
    encrypted_file = encrypt_with_fernet(file_contents, aes_key)
    print("Encrypted File:", encrypted_file)

    # Encrypt AES key with RSA public key
    encrypted_aes_key = encrypt_with_rsa(public_key, aes_key)
    print("Encrypted AES Key:", encrypted_aes_key)

    # Demonstrate decryption of the AES key using the RSA private key
    decrypted_aes_key = decrypt_with_rsa(private_key, encrypted_aes_key)
    print("Decrypted AES Key:", decrypted_aes_key)

    # Decrypt the encrypted file using the decrypted AES key
    fernet = Fernet(decrypted_aes_key)
    decrypted_file = fernet.decrypt(encrypted_file)
    print("Decrypted File:", decrypted_file)
