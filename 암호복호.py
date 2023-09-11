from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

def encrypt(plaintext, key):
    backend = default_backend()

    # 키를 바이트로 변환
    key = bytes.fromhex(key)

    # 16바이트로 패딩
    padder = padding.PKCS7(128).padder()
    padded_plaintext = padder.update(plaintext.encode()) + padder.finalize()

    # AES 암호화
    iv = b'0123456789abcdef'  # 16바이트 초기화 벡터 (IV)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=backend)
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()

    # Base64 인코딩하여 출력
    return base64.b64encode(ciphertext).decode()

def decrypt(ciphertext, key):
    backend = default_backend()

    # 키를 바이트로 변환
    key = bytes.fromhex(key)

    # Base64 디코딩
    ciphertext = base64.b64decode(ciphertext)

    # AES 복호화
    iv = b'0123456789abcdef'  # 16바이트 초기화 벡터 (IV)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=backend)
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # 패딩 제거
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()

    return plaintext.decode()

# 테스트
key = "0123456789abcdef"  # 16바이트 키
plaintext = "Hello, World! This is a test message."

encrypted_text = encrypt(plaintext, key)
print("암호화된 데이터:", encrypted_text)

decrypted_text = decrypt(encrypted_text, key)
print("복호화된 데이터:", decrypted_text)