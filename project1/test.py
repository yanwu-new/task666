from sm4 import encrypt_ecb, decrypt_ecb

key = b"Sixteen byte key"          # 16 字节
msg = b"Hello, SM4 in pure Python!"

cipher = encrypt_ecb(msg, key)
plain  = decrypt_ecb(cipher, key)

print("密文:", cipher.hex())
print("解密结果:", plain)          # 与原文相同
