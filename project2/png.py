import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import random

class DigitalWatermark:
    def __init__(self, alpha=0.1, wavelet='db4'):
        """
        Digital watermark class
        
        Parameters:
        alpha: Watermark strength coefficient
        wavelet: Wavelet transform type
        """
        self.alpha = alpha
        self.wavelet = wavelet
    
    def embed_watermark_dct(self, host_image, watermark_image):
        """
        DCT-based watermark embedding
        """
        # Convert to grayscale
        if len(host_image.shape) == 3:
            host_gray = cv2.cvtColor(host_image, cv2.COLOR_BGR2GRAY)
        else:
            host_gray = host_image.copy()
        
        if len(watermark_image.shape) == 3:
            watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)
        else:
            watermark_gray = watermark_image.copy()
        
        # Resize watermark
        h, w = host_gray.shape
        watermark_resized = cv2.resize(watermark_gray, (w//8, h//8))
        
        # Block DCT transform
        watermarked = host_gray.copy().astype(np.float32)
        wm_flat = watermark_resized.flatten()
        
        # 8x8 block processing
        for i in range(0, h-7, 8):
            for j in range(0, w-7, 8):
                block = watermarked[i:i+8, j:j+8]
                
                # DCT transform
                dct_block = cv2.dct(block)
                
                # Embed watermark in mid-frequency components
                if len(wm_flat) > 0:
                    wm_bit = wm_flat[((i//8) * (w//8) + (j//8)) % len(wm_flat)]
                    dct_block[2, 2] += self.alpha * wm_bit
                
                # Inverse DCT
                watermarked[i:i+8, j:j+8] = cv2.idct(dct_block)
        
        return np.clip(watermarked, 0, 255).astype(np.uint8)
    
    def extract_watermark_dct(self, original_image, watermarked_image, watermark_shape):
        """
        DCT-based watermark extraction
        """
        # Convert to grayscale
        if len(original_image.shape) == 3:
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_image.copy()
        
        if len(watermarked_image.shape) == 3:
            wm_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
        else:
            wm_gray = watermarked_image.copy()
        
        orig_gray = orig_gray.astype(np.float32)
        wm_gray = wm_gray.astype(np.float32)
        
        h, w = orig_gray.shape
        extracted_wm = []
        
        # 8x8 block processing
        for i in range(0, h-7, 8):
            for j in range(0, w-7, 8):
                orig_block = orig_gray[i:i+8, j:j+8]
                wm_block = wm_gray[i:i+8, j:j+8]
                
                # DCT transform
                orig_dct = cv2.dct(orig_block)
                wm_dct = cv2.dct(wm_block)
                
                # Extract watermark
                extracted_val = (wm_dct[2, 2] - orig_dct[2, 2]) / self.alpha
                extracted_wm.append(extracted_val)
        
        # Reconstruct watermark
        wm_size = watermark_shape[0] * watermark_shape[1]
        if len(extracted_wm) >= wm_size:
            extracted_wm = extracted_wm[:wm_size]
            extracted_watermark = np.array(extracted_wm).reshape(watermark_shape)
            return np.clip(extracted_watermark, 0, 255).astype(np.uint8)
        else:
            return np.zeros(watermark_shape, dtype=np.uint8)
    
    def embed_watermark_dwt(self, host_image, watermark_image):
        """
        DWT-based watermark embedding
        """
        # Convert to grayscale
        if len(host_image.shape) == 3:
            host_gray = cv2.cvtColor(host_image, cv2.COLOR_BGR2GRAY)
        else:
            host_gray = host_image.copy()
        
        if len(watermark_image.shape) == 3:
            watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)
        else:
            watermark_gray = watermark_image.copy()
        
        # Wavelet transform
        coeffs = pywt.dwt2(host_gray, self.wavelet)
        LL, (LH, HL, HH) = coeffs
        
        # Resize watermark according to LL subband
        ll_h, ll_w = LL.shape
        watermark_resized = cv2.resize(watermark_gray, (ll_w, ll_h))
        
        # Embed watermark in LL subband
        LL_watermarked = LL + self.alpha * watermark_resized
        
        # Inverse wavelet transform
        watermarked_coeffs = (LL_watermarked, (LH, HL, HH))
        watermarked = pywt.idwt2(watermarked_coeffs, self.wavelet)
        
        return np.clip(watermarked, 0, 255).astype(np.uint8)
    
    def extract_watermark_dwt(self, original_image, watermarked_image):
        """
        DWT-based watermark extraction
        """
        # Convert to grayscale
        if len(original_image.shape) == 3:
            orig_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_image.copy()
        
        if len(watermarked_image.shape) == 3:
            wm_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
        else:
            wm_gray = watermarked_image.copy()
        
        # Ensure images have same dimensions
        if orig_gray.shape != wm_gray.shape:
            wm_gray = cv2.resize(wm_gray, (orig_gray.shape[1], orig_gray.shape[0]))
        
        # Wavelet transform
        orig_coeffs = pywt.dwt2(orig_gray, self.wavelet)
        wm_coeffs = pywt.dwt2(wm_gray, self.wavelet)
        
        orig_LL, _ = orig_coeffs
        wm_LL, _ = wm_coeffs
        
        # Ensure LL subbands have same dimensions
        if orig_LL.shape != wm_LL.shape:
            wm_LL = cv2.resize(wm_LL, (orig_LL.shape[1], orig_LL.shape[0]))
        
        # Extract watermark
        extracted_watermark = (wm_LL - orig_LL) / self.alpha
        
        return np.clip(extracted_watermark, 0, 255).astype(np.uint8)

class RobustnessTest:
    def __init__(self):
        """Robustness test class"""
        pass
    
    def flip_image(self, image, direction='horizontal'):
        """Image flipping"""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        else:
            return cv2.flip(image, -1)  # Both horizontal and vertical
    
    def translate_image(self, image, tx=10, ty=10):
        """Image translation"""
        rows, cols = image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (cols, rows))
    
    def crop_image(self, image, crop_ratio=0.8):
        """Image cropping"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        return image[start_h:start_h+new_h, start_w:start_w+new_w]
    
    def adjust_contrast(self, image, factor=1.5):
        """Contrast adjustment"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    def adjust_brightness(self, image, factor=1.2):
        """Brightness adjustment"""
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pil_image)
        enhanced = enhancer.enhance(factor)
        return np.array(enhanced)
    
    def add_noise(self, image, noise_type='gaussian', noise_level=0.05):
        """Add noise to image"""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level * 255, image.shape)
            noisy_image = image + noise
        elif noise_type == 'salt_pepper':
            noisy_image = image.copy()
            salt_pepper_ratio = noise_level
            num_salt = int(salt_pepper_ratio * image.size * 0.5)
            num_pepper = int(salt_pepper_ratio * image.size * 0.5)
            
            # Add salt noise
            coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
            noisy_image[coords[0], coords[1]] = 255
            
            # Add pepper noise
            coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
            noisy_image[coords[0], coords[1]] = 0
        else:
            return image
        
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def rotate_image(self, image, angle=15):
        """Image rotation"""
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))
    
    def compress_image(self, image, quality=50):
        """JPEG compression"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

def calculate_metrics(original, extracted):
    """Calculate image quality metrics"""
    # Ensure images have same dimensions
    if original.shape != extracted.shape:
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))
    
    # Convert to float for calculation
    original_float = original.astype(np.float64)
    extracted_float = extracted.astype(np.float64)
    
    # Calculate PSNR
    mse = np.mean((original_float - extracted_float) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate SSIM
    try:
        ssim_value = ssim(original, extracted, data_range=255)
    except:
        ssim_value = 0.0
    
    # Calculate correlation coefficient
    try:
        correlation = np.corrcoef(original_float.flatten(), extracted_float.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    except:
        correlation = 0.0
    
    return psnr, ssim_value, correlation

def comprehensive_test():
    """Comprehensive test function"""
    # Create test images
    host_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    watermark_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    
    # Initialize watermark system
    watermark_system = DigitalWatermark(alpha=0.1)
    robustness_test = RobustnessTest()
    
    # Embed watermarks
    print("Embedding watermarks...")
    watermarked_dct = watermark_system.embed_watermark_dct(host_image, watermark_image)
    watermarked_dwt = watermark_system.embed_watermark_dwt(host_image, watermark_image)
    
    # Robustness tests
    test_operations = [
        ("Original", lambda x: x),
        ("Horizontal flip", lambda x: robustness_test.flip_image(x, 'horizontal')),
        ("Vertical flip", lambda x: robustness_test.flip_image(x, 'vertical')),
        ("Translation", lambda x: robustness_test.translate_image(x, 20, 20)),
        ("Cropping", lambda x: robustness_test.crop_image(x, 0.8)),
        ("Contrast adjust", lambda x: robustness_test.adjust_contrast(x, 1.5)),
        ("Brightness adjust", lambda x: robustness_test.adjust_brightness(x, 1.2)),
        ("Gaussian noise", lambda x: robustness_test.add_noise(x, 'gaussian', 0.05)),
        ("Salt-pepper noise", lambda x: robustness_test.add_noise(x, 'salt_pepper', 0.05)),
        ("Rotation", lambda x: robustness_test.rotate_image(x, 15)),
        ("JPEG compression", lambda x: robustness_test.compress_image(x, 70))
    ]
    
    print("\nRobustness test results:")
    print("=" * 80)
    print(f"{'Attack Type':<20} {'Method':<6} {'PSNR':<10} {'SSIM':<10} {'Correlation':<10}")
    print("-" * 80)
    
    for test_name, test_func in test_operations:
        # DCT method test
        attacked_image = test_func(watermarked_dct)
        if attacked_image.shape != host_image.shape:
            attacked_image = cv2.resize(attacked_image, (host_image.shape[1], host_image.shape[0]))
        
        extracted_wm_dct = watermark_system.extract_watermark_dct(
            host_image, attacked_image, watermark_image.shape
        )
        
        psnr_dct, ssim_dct, corr_dct = calculate_metrics(watermark_image, extracted_wm_dct)
        
        # DWT method test
        attacked_image_dwt = test_func(watermarked_dwt)
        if attacked_image_dwt.shape != host_image.shape:
            attacked_image_dwt = cv2.resize(attacked_image_dwt, (host_image.shape[1], host_image.shape[0]))
        
        extracted_wm_dwt = watermark_system.extract_watermark_dwt(host_image, attacked_image_dwt)
        
        # Resize extracted watermark to match original
        if extracted_wm_dwt.shape != watermark_image.shape:
            extracted_wm_dwt = cv2.resize(extracted_wm_dwt, (watermark_image.shape[1], watermark_image.shape[0]))
        
        psnr_dwt, ssim_dwt, corr_dwt = calculate_metrics(watermark_image, extracted_wm_dwt)
        
        print(f"{test_name:<20} {'DCT':<6} {psnr_dct:<10.2f} {ssim_dct:<10.4f} {corr_dct:<10.4f}")
        print(f"{'':<20} {'DWT':<6} {psnr_dwt:<10.2f} {ssim_dwt:<10.4f} {corr_dwt:<10.4f}")
        print("-" * 80)

def demo_with_real_images():
    """Demo with synthetic images"""
    print("Creating demo images...")
    
    # Create host image (simplified Lena pattern)
    host = np.zeros((512, 512), dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            host[i, j] = int(128 + 100 * np.sin(i/20) * np.cos(j/20))
    
    # Create watermark image
    watermark = np.zeros((64, 64), dtype=np.uint8)
    cv2.rectangle(watermark, (10, 10), (54, 54), 255, -1)
    cv2.rectangle(watermark, (20, 20), (44, 44), 0, -1)
    cv2.putText(watermark, 'WM', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    
    # Initialize watermark system
    watermark_system = DigitalWatermark(alpha=0.15)
    
    # Embed watermark
    watermarked_image = watermark_system.embed_watermark_dct(host, watermark)
    
    # Extract watermark
    extracted_watermark = watermark_system.extract_watermark_dct(
        host, watermarked_image, watermark.shape
    )
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(host, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(watermark, cmap='gray')
    plt.title('Watermark Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(watermarked_image, cmap='gray')
    plt.title('Watermarked Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(extracted_watermark, cmap='gray')
    plt.title('Extracted Watermark')
    plt.axis('off')
    
    # Calculate quality metrics
    psnr, ssim_val, corr = calculate_metrics(watermark, extracted_watermark)
    
    plt.subplot(2, 3, 5)
    plt.text(0.1, 0.8, f'PSNR: {psnr:.2f} dB', fontsize=12)
    plt.text(0.1, 0.6, f'SSIM: {ssim_val:.4f}', fontsize=12)
    plt.text(0.1, 0.4, f'Correlation: {corr:.4f}', fontsize=12)
    plt.text(0.1, 0.2, f'Alpha: {watermark_system.alpha}', fontsize=12)
    plt.title('Quality Metrics')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nImage quality metrics:")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"Correlation: {corr:.4f}")

if __name__ == "__main__":
    print("Digital Watermark System Test")
    print("=" * 50)
    
    # Run comprehensive test
    comprehensive_test()
    
    # Run demo
    demo_with_real_images()
    
    print("\nTesting completed!")