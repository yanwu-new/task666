#!/usr/bin/env python3
# optimized_sm4.py — 根据PPT优化思路改进的SM4实现

########################################
# 1. SM4 常量
########################################
SBOX = [
    0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05,
    0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99,
    0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62,
    0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6,
    0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8,
    0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35,
    0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87,
    0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e,
    0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1,
    0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3,
    0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f,
    0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51,
    0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8,
    0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0,
    0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84,
    0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48
]

FK = [0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc]

CK = [
    0x00070e15,0x1c232a31,0x383f464d,0x545b6269,
    0x70777e85,0x8c939aa1,0xa8afb6bd,0xc4cbd2d9,
    0xe0e7eef5,0xfc030a11,0x181f262d,0x343b4249,
    0x50575e65,0x6c737a81,0x888f969d,0xa4abb2b9,
    0xc0c7ced5,0xdce3eaf1,0xf8ff060d,0x141b2229,
    0x30373e45,0x4c535a61,0x686f767d,0x848b9299,
    0xa0a7aeb5,0xbcc3cad1,0xd8dfe6ed,0xf4fb0209,
    0x10171e25,0x2c333a41,0x484f565d,0x646b7279
]

########################################
# 2. T-Table 优化：预计算SBox+线性变换
########################################
def _generate_t_tables():
    """生成4个T-Table，将SBox和线性变换合并"""
    def _rotl32(x: int, n: int) -> int:
        n &= 31
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF
    
    def _L(b: int) -> int:
        return b ^ _rotl32(b, 2) ^ _rotl32(b, 10) ^ _rotl32(b, 18) ^ _rotl32(b, 24)
    
    # 4个T-Table，每个256个32位元素
    T0 = [0] * 256
    T1 = [0] * 256  
    T2 = [0] * 256
    T3 = [0] * 256
    
    for i in range(256):
        sbox_val = SBOX[i]
        # T0: SBox值在最高字节位置
        temp = sbox_val << 24
        T0[i] = _L(temp)
        
        # T1: SBox值在次高字节位置
        temp = sbox_val << 16
        T1[i] = _L(temp)
        
        # T2: SBox值在次低字节位置
        temp = sbox_val << 8
        T2[i] = _L(temp)
        
        # T3: SBox值在最低字节位置
        temp = sbox_val
        T3[i] = _L(temp)
    
    return T0, T1, T2, T3

# 全局预计算T-Table
T0, T1, T2, T3 = _generate_t_tables()

########################################
# 3. 优化后的基础函数
########################################
def _rotl32(x: int, n: int) -> int:
    """32‑bit 循环左移"""
    n &= 31
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF

def _tau(a: int) -> int:
    """字节代换（原版，用于密钥扩展）"""
    return (
        (SBOX[(a >> 24) & 0xFF] << 24) |
        (SBOX[(a >> 16) & 0xFF] << 16) |
        (SBOX[(a >> 8)  & 0xFF] << 8)  |
        (SBOX[a & 0xFF])
    )

def _L_prime(b: int) -> int:
    """密钥扩展时线性变换"""
    return b ^ _rotl32(b, 13) ^ _rotl32(b, 23)

def _T_prime(x: int) -> int:
    """T' 变换（密钥扩展）"""
    return _L_prime(_tau(x))

def _T_optimized(x: int) -> int:
    """优化的T变换：使用T-Table直接计算"""
    return (T0[(x >> 24) & 0xFF] ^ 
            T1[(x >> 16) & 0xFF] ^ 
            T2[(x >> 8)  & 0xFF] ^ 
            T3[x & 0xFF])

########################################
# 4. 密钥扩展
########################################
def key_schedule(key: bytes) -> list[int]:
    """生成 32 个轮密钥 rk[0..31]"""
    if len(key) != 16:
        raise ValueError("SM4 密钥必须为 16 字节 (128 bit)")
    MK = [int.from_bytes(key[i*4:(i+1)*4], 'big') for i in range(4)]
    K  = [MK[i] ^ FK[i] for i in range(4)]
    rk = []
    for i in range(32):
        new_k = K[i] ^ _T_prime(K[i+1] ^ K[i+2] ^ K[i+3] ^ CK[i])
        K.append(new_k)
        rk.append(new_k)
    return rk

########################################
# 5. 优化的单块加/解密 - 循环展开
########################################
def encrypt_block_optimized(block16: bytes, rk: list[int]) -> bytes:
    """优化的加密：T-Table + 循环展开"""
    if len(block16) != 16:
        raise ValueError("明文块必须为 16 字节")
    
    # 初始化4个32位字
    x0 = int.from_bytes(block16[0:4], 'big')
    x1 = int.from_bytes(block16[4:8], 'big')
    x2 = int.from_bytes(block16[8:12], 'big')
    x3 = int.from_bytes(block16[12:16], 'big')
    
    # 32轮展开，每4轮一组
    for i in range(0, 32, 4):
        # 第i轮
        tmp = x1 ^ x2 ^ x3 ^ rk[i]
        x0 ^= _T_optimized(tmp)
        
        # 第i+1轮
        tmp = x2 ^ x3 ^ x0 ^ rk[i+1]
        x1 ^= _T_optimized(tmp)
        
        # 第i+2轮
        tmp = x3 ^ x0 ^ x1 ^ rk[i+2]
        x2 ^= _T_optimized(tmp)
        
        # 第i+3轮
        tmp = x0 ^ x1 ^ x2 ^ rk[i+3]
        x3 ^= _T_optimized(tmp)
    
    # 输出时逆序
    return (x3.to_bytes(4, 'big') + x2.to_bytes(4, 'big') + 
            x1.to_bytes(4, 'big') + x0.to_bytes(4, 'big'))

def decrypt_block_optimized(block16: bytes, rk: list[int]) -> bytes:
    """优化的解密：轮密钥反序"""
    return encrypt_block_optimized(block16, rk[::-1])

########################################
# 6. 批量处理优化
########################################
def encrypt_blocks_batch(blocks: list[bytes], rk: list[int]) -> list[bytes]:
    """批量加密多个块，减少函数调用开销"""
    return [encrypt_block_optimized(block, rk) for block in blocks]

def decrypt_blocks_batch(blocks: list[bytes], rk: list[int]) -> list[bytes]:
    """批量解密多个块"""
    return [decrypt_block_optimized(block, rk) for block in blocks]

########################################
# 7. 兼容原接口
########################################
def encrypt_block(block16: bytes, rk: list[int]) -> bytes:
    """兼容原接口的加密函数"""
    return encrypt_block_optimized(block16, rk)

def decrypt_block(block16: bytes, rk: list[int]) -> bytes:
    """兼容原接口的解密函数"""
    return decrypt_block_optimized(block16, rk)

########################################
# 8. ECB模式（保持不变）
########################################
def _pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len] * pad_len)

def _pkcs7_unpad(padded: bytes) -> bytes:
    if not padded:
        raise ValueError("数据为空")
    pad_len = padded[-1]
    if pad_len < 1 or pad_len > 16 or padded[-pad_len:] != bytes([pad_len] * pad_len):
        raise ValueError("PKCS#7 填充无效")
    return padded[:-pad_len]

def encrypt_ecb(plaintext: bytes, key: bytes) -> bytes:
    """优化的ECB加密"""
    rk = key_schedule(key)
    padded = _pkcs7_pad(plaintext)
    
    # 批量处理
    blocks = [padded[i:i+16] for i in range(0, len(padded), 16)]
    encrypted_blocks = encrypt_blocks_batch(blocks, rk)
    return b''.join(encrypted_blocks)

def decrypt_ecb(ciphertext: bytes, key: bytes) -> bytes:
    """优化的ECB解密"""
    if len(ciphertext) % 16:
        raise ValueError("密文长度应为 16 的倍数")
    
    rk = key_schedule(key)
    blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
    decrypted_blocks = decrypt_blocks_batch(blocks, rk)
    plain_padded = b''.join(decrypted_blocks)
    return _pkcs7_unpad(plain_padded)

########################################
# 9. 性能测试函数
########################################
def benchmark_comparison(test_data: bytes, key: bytes, iterations: int = 1000):
    """性能对比测试"""
    import time
    
    rk = key_schedule(key)
    
    # 原始版本（模拟）
    def encrypt_block_original(block16: bytes, rk: list[int]) -> bytes:
        X = [int.from_bytes(block16[i*4:(i+1)*4], 'big') for i in range(4)]
        for i in range(32):
            tmp = X[i+1] ^ X[i+2] ^ X[i+3] ^ rk[i]
            # 原始T变换（分离的tau和L）
            b = ((SBOX[(tmp >> 24) & 0xFF] << 24) |
                 (SBOX[(tmp >> 16) & 0xFF] << 16) |
                 (SBOX[(tmp >> 8)  & 0xFF] << 8)  |
                 (SBOX[tmp & 0xFF]))
            t_result = b ^ _rotl32(b, 2) ^ _rotl32(b, 10) ^ _rotl32(b, 18) ^ _rotl32(b, 24)
            X.append(X[i] ^ t_result)
        return b''.join(X[i].to_bytes(4, 'big') for i in (35,34,33,32))
    
    # 测试原始版本
    start = time.time()
    for _ in range(iterations):
        encrypt_block_original(test_data, rk)
    original_time = time.time() - start
    
    # 测试优化版本
    start = time.time()
    for _ in range(iterations):
        encrypt_block_optimized(test_data, rk)
    optimized_time = time.time() - start
    
    speedup = original_time / optimized_time
    print(f"原始版本: {original_time:.4f}秒")
    print(f"优化版本: {optimized_time:.4f}秒")
    print(f"性能提升: {speedup:.2f}x")
    return speedup

########################################
# 10. 自测向量
########################################
if __name__ == "__main__":
    print("=" * 60)
    print("SM4 优化前后对比测试")
    print("=" * 60)
    
    # 测试数据
    key = bytes.fromhex("0123456789abcdeffedcba9876543210")
    data = bytes.fromhex("0123456789abcdeffedcba9876543210")
    
    print(f"测试密钥: {key.hex().upper()}")
    print(f"测试明文: {data.hex().upper()}")
    print(f"明文文本: {data}")
    print()
    
    # 生成轮密钥
    rk_list = key_schedule(key)
    
    # 原始实现（用于对比）
    def encrypt_block_original(block16: bytes, rk: list[int]) -> bytes:
        """原始实现版本"""
        X = [int.from_bytes(block16[i*4:(i+1)*4], 'big') for i in range(4)]
        for i in range(32):
            tmp = X[i+1] ^ X[i+2] ^ X[i+3] ^ rk[i]
            # 原始T变换（分离的tau和L）
            b = ((SBOX[(tmp >> 24) & 0xFF] << 24) |
                 (SBOX[(tmp >> 16) & 0xFF] << 16) |
                 (SBOX[(tmp >> 8)  & 0xFF] << 8)  |
                 (SBOX[tmp & 0xFF]))
            t_result = b ^ _rotl32(b, 2) ^ _rotl32(b, 10) ^ _rotl32(b, 18) ^ _rotl32(b, 24)
            X.append(X[i] ^ t_result)
        return b''.join(X[i].to_bytes(4, 'big') for i in (35,34,33,32))
    
    def decrypt_block_original(block16: bytes, rk: list[int]) -> bytes:
        """原始解密实现"""
        return encrypt_block_original(block16, rk[::-1])
    
    # 单块加密对比
    print("=" * 30 + " 单块加密对比 " + "=" * 30)
    
    # 原始版本
    cipher_original = encrypt_block_original(data, rk_list)
    decrypted_original = decrypt_block_original(cipher_original, rk_list)
    
    print("【原始实现】")
    print(f"  加密结果: {cipher_original.hex().upper()}")
    print(f"  解密结果: {decrypted_original.hex().upper()}")
    print(f"  解密文本: {decrypted_original}")
    print(f"  正确性检查: {'✓ 通过' if decrypted_original == data else '✗ 失败'}")
    print()
    
    # 优化版本
    cipher_optimized = encrypt_block_optimized(data, rk_list)
    decrypted_optimized = decrypt_block_optimized(cipher_optimized, rk_list)
    
    print("【优化实现】")
    print(f"  加密结果: {cipher_optimized.hex().upper()}")
    print(f"  解密结果: {decrypted_optimized.hex().upper()}")
    print(f"  解密文本: {decrypted_optimized}")
    print(f"  正确性检查: {'✓ 通过' if decrypted_optimized == data else '✗ 失败'}")
    print()
    
    # 验证两种实现结果一致
    print("【一致性验证】")
    encryption_match = cipher_original == cipher_optimized
    decryption_match = decrypted_original == decrypted_optimized
    print(f"  加密结果一致: {'✓ 是' if encryption_match else '✗ 否'}")
    print(f"  解密结果一致: {'✓ 是' if decryption_match else '✗ 否'}")
    print(f"  标准向量验证: {'✓ 通过' if cipher_optimized.hex() == '681edf34d206965e86b3e94f536e4246' else '✗ 失败'}")
    print()
    
    # 长文本测试
    print("=" * 30 + " 长文本ECB测试 " + "=" * 30)
    
    long_plaintext = b"This is a longer test message for SM4 ECB mode encryption and decryption testing. Hello World!"
    print(f"长文本明文: {long_plaintext}")
    print(f"明文长度: {len(long_plaintext)} 字节")
    print()
    
    # 原始ECB实现（用于对比）
    def encrypt_ecb_original(plaintext: bytes, key: bytes) -> bytes:
        rk = key_schedule(key)
        padded = _pkcs7_pad(plaintext)
        return b''.join(encrypt_block_original(padded[i:i+16], rk) for i in range(0, len(padded), 16))
    
    def decrypt_ecb_original(ciphertext: bytes, key: bytes) -> bytes:
        if len(ciphertext) % 16:
            raise ValueError("密文长度应为 16 的倍数")
        rk = key_schedule(key)
        plain_padded = b''.join(decrypt_block_original(ciphertext[i:i+16], rk) for i in range(0, len(ciphertext), 16))
        return _pkcs7_unpad(plain_padded)
    
    # 原始ECB
    ciphertext_original = encrypt_ecb_original(long_plaintext, key)
    recovered_original = decrypt_ecb_original(ciphertext_original, key)
    
    print("【原始ECB实现】")
    print(f"  密文: {ciphertext_original.hex().upper()}")
    print(f"  密文长度: {len(ciphertext_original)} 字节")
    print(f"  解密结果: {recovered_original}")
    print(f"  正确性检查: {'✓ 通过' if recovered_original == long_plaintext else '✗ 失败'}")
    print()
    
    # 优化ECB
    ciphertext_optimized = encrypt_ecb(long_plaintext, key)
    recovered_optimized = decrypt_ecb(ciphertext_optimized, key)
    
    print("【优化ECB实现】")
    print(f"  密文: {ciphertext_optimized.hex().upper()}")
    print(f"  密文长度: {len(ciphertext_optimized)} 字节")
    print(f"  解密结果: {recovered_optimized}")
    print(f"  正确性检查: {'✓ 通过' if recovered_optimized == long_plaintext else '✗ 失败'}")
    print()
    
    # ECB一致性验证
    print("【ECB一致性验证】")
    ecb_encryption_match = ciphertext_original == ciphertext_optimized
    ecb_decryption_match = recovered_original == recovered_optimized
    print(f"  ECB加密结果一致: {'✓ 是' if ecb_encryption_match else '✗ 否'}")
    print(f"  ECB解密结果一致: {'✓ 是' if ecb_decryption_match else '✗ 否'}")
    print()
    
    # 批量测试
    print("=" * 30 + " 批量处理测试 " + "=" * 30)
    test_blocks = [data] * 4
    print(f"测试块数: {len(test_blocks)}")
    print(f"每块内容: {data.hex().upper()}")
    
    encrypted_batch = encrypt_blocks_batch(test_blocks, rk_list)
    decrypted_batch = decrypt_blocks_batch(encrypted_batch, rk_list)
    
    print("【批量加密结果】")
    for i, block in enumerate(encrypted_batch):
        print(f"  块 {i+1}: {block.hex().upper()}")
    
    batch_success = all(block == data for block in decrypted_batch)
    print(f"  批量解密验证: {'✓ 通过' if batch_success else '✗ 失败'}")
    print()
    
    # 性能对比
    print("=" * 30 + " 性能对比测试 " + "=" * 30)
    speedup = benchmark_comparison(data, key, 1000)
    print(f"T-Table优化带来约 {speedup:.1f}x 性能提升")
    print()
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_tests_passed = (
        encryption_match and decryption_match and 
        ecb_encryption_match and ecb_decryption_match and 
        batch_success and
        cipher_optimized.hex() == "681edf34d206965e86b3e94f536e4246"
    )
    
    print(f"✓ 单块加密/解密一致性: {'通过' if encryption_match and decryption_match else '失败'}")
    print(f"✓ ECB模式一致性: {'通过' if ecb_encryption_match and ecb_decryption_match else '失败'}")
    print(f"✓ 批量处理: {'通过' if batch_success else '失败'}")
    print(f"✓ 标准向量验证: {'通过' if cipher_optimized.hex() == '681edf34d206965e86b3e94f536e4246' else '失败'}")
    print(f"✓ 性能提升: {speedup:.1f}x")
    print()
    
    if all_tests_passed:
        print("🎉 所有测试通过！优化版本功能正确且性能提升显著！")
    else:
        print("❌ 部分测试失败，请检查实现！")
    
    print("=" * 60)