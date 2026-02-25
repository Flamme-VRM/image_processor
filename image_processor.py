#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREMIUM IMAGE PROCESSOR v1.0
Watermark Removal + AI Upscaling Pipeline
For: Premium Scrollytelling Website Assets

Pipeline:
  1. Extract .jpg images from a .zip archive
  2. Remove watermarks via OpenCV inpainting (mask-based or auto-detect)
  3. AI-upscale 2x-4x via Real-ESRGAN (sharp, no blur)
  4. Save results as high-quality PNG or JPG
"""

import os
import sys
import hashlib
import zipfile
import shutil
import tempfile
import argparse
import time
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np

import cv2
from PIL import Image, ImageFilter
try:
    import redis as redis_lib
except ImportError:
    redis_lib = None

#  Logging Setup 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ImageProcessor")


def is_valid_img(filepath):
    try:
        img = cv2.imread(filepath)
        return img is not None and img.size > 0
    except:
        return False

class ZipExtractor:
    """Safely extract .jpg files from a .zip archive."""

    JUNK_FILES = {".DS_Store", "Thumbs.db", "__MACOSX", ".gitkeep", "desktop.ini"}
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(self, zip_path: str, temp_dir: str):
        self.zip_path = Path(zip_path)
        self.temp_dir = Path(temp_dir)

        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP archive not found: {self.zip_path}")
        if not zipfile.is_zipfile(str(self.zip_path)):
            raise ValueError(f"File is not a valid ZIP archive: {self.zip_path}")

    def extract(self) -> List[str]:
        """Extract valid image files, return list of absolute paths."""
        extracted = []
        with zipfile.ZipFile(str(self.zip_path), 'r') as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)

                # Skip directories, junk files, hidden files
                if not basename:
                    continue
                if basename in self.JUNK_FILES:
                    continue
                if basename.startswith(".") or basename.startswith("__"):
                    continue

                ext = os.path.splitext(basename)[1].lower()
                if ext not in self.VALID_EXTENSIONS:
                    continue

                # Extract to flat directory (avoid nested folder issues)
                target_path = self.temp_dir / basename
                # Handle duplicate names
                counter = 1
                while target_path.exists():
                    stem = Path(basename).stem
                    target_path = self.temp_dir / f"{stem}_{counter}{ext}"
                    counter += 1

                with zf.open(member) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

                extracted.append(str(target_path))

        log.info("Extracted %d images from archive", len(extracted))
        return extracted


# Watermark removal

class WatermarkRemover:
    """
    Remove watermarks using OpenCV inpainting.
    
    Supports two modes:
      A) MASK MODE — provide a binary mask image (white = watermark area)
         Best for static watermarks in the same position on every image.
      B) AUTO-DETECT MODE — automatically detect semi-transparent bright
         overlays typical for stock photo watermarks.
    """

    def __init__(self, mask_path: Optional[str] = None,
                 inpaint_radius: int = 7,
                 method: str = "telea",
                 auto_threshold: int = 200,
                 auto_min_area: int = 500):
        """
        Args:
            mask_path: Path to a binary mask image (white=watermark). None = auto-detect.
            inpaint_radius: Radius for cv2.inpaint (pixels). Larger = smoother fill.
            method: "telea" (fast, good) or "ns" (Navier-Stokes, slower, sometimes better).
            auto_threshold: Brightness threshold for auto-detection (0-255).
            auto_min_area: Minimum contour area to consider as watermark (pixels²).
        """
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = (cv2.INPAINT_TELEA if method == "telea"
                               else cv2.INPAINT_NS)
        self.auto_threshold = auto_threshold
        self.auto_min_area = auto_min_area

        self.static_mask = None
        if mask_path and os.path.exists(mask_path):
            self.static_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if self.static_mask is not None:
                log.info(f"Using static watermark mask: {mask_path}")
            else:
                log.warning(f"Could not load mask: {mask_path}, falling back to auto-detect")

    def _auto_detect_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-detect watermark regions.
        
        Specifically tailored for faint app logos (like 'Veo') in the extreme
        bottom-right corner. We use a relaxed brightness threshold but restrict
        the search to a very small region to avoid erasing image content.
        """
        h, w = image.shape[:2]
        
        # Veo logo usually occupies roughly the last 5-7% of width and 5% of height
        x_start = int(w * 0.93)
        y_start = int(h * 0.94)
        
        # Ensure we don't go out of bounds
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        
        roi = image[y_start:h, x_start:w]
        if roi.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Relaxed threshold for faint thin text like 'Veo' to ensure full coverage
        bright = hsv_roi[:, :, 2] > 120
        gray = hsv_roi[:, :, 1] < 100
        
        mask_roi = np.zeros_like(hsv_roi[:, :, 2])
        mask_roi[np.logical_and(bright, gray)] = 255
        
        # Heavy dilation to merge thin letters into a solid block for clean inpainting
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_roi = cv2.dilate(mask_roi, dilate_kernel, iterations=2)
        
        # Place ROI mask back into the full-size mask
        final_mask = np.zeros((h, w), dtype=np.uint8)
        final_mask[y_start:h, x_start:w] = mask_roi
        
        return final_mask

    def remove(self, image: np.ndarray) -> np.ndarray:
        """Remove watermark from an image using inpainting."""
        if self.static_mask is not None:
            mask = self.static_mask.copy()
            # Resize mask to match image if needed
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            mask = self._auto_detect_mask(image)

        # Check if any watermark was detected
        if np.count_nonzero(mask) == 0:
            return image  # No watermark found

        # Apply inpainting
        result = cv2.inpaint(image, mask, self.inpaint_radius, self.inpaint_method)
        return result


# Upscaling classes

class AIUpscaler:
    """
    AI-based image upscaling. Tries (in order):
      1. Real-ESRGAN (pip install realesrgan) — best quality
      2. OpenCV DNN Super Resolution (EDSR model) — good fallback
      3. PIL Lanczos + Unsharp Mask — last resort (still better than raw resize)
    """

    def __init__(self, scale: int = 4, gpu: bool = True):
        """
        Args:
            scale: Upscale factor (2 or 4).
            gpu: Use GPU if available (CUDA for Real-ESRGAN).
        """
        self.scale = scale
        self.gpu = gpu
        self.method_name = "unknown"
        self._upscaler = None

        # Try to initialize the best available method
        self._init_realesrgan() or self._init_opencv_dnn() or self._init_pillow()
        log.info(f"Upscaler initialized: {self.method_name} (scale={self.scale}x)")

    def _init_realesrgan(self) -> bool:
        """Try to initialize Real-ESRGAN."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            # Select model based on scale
            if self.scale == 2:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=2)
                model_name = "RealESRGAN_x2plus"
                netscale = 2
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                num_block=23, num_grow_ch=32, scale=4)
                model_name = "RealESRGAN_x4plus"
                netscale = 4

            # Look for model weights in several locations
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f"{model_name}.pth"

            if not model_path.exists():
                # Try to auto-download
                log.info(f" Downloading {model_name} model weights...")
                try:
                    import urllib.request
                    url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth"
                    if self.scale == 2:
                        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{model_name}.pth"
                    urllib.request.urlretrieve(url, str(model_path))
                    log.info(f"Model downloaded to {model_path}")
                except Exception as e:
                    log.warning(f"Auto-download failed: {e}")
                    log.warning(f"Please manually download {model_name}.pth to {model_dir}/")
                    return False

            self._upscaler = RealESRGANer(
                scale=netscale,
                model_path=str(model_path),
                model=model,
                tile=512,          # Process in tiles to save VRAM
                tile_pad=10,
                pre_pad=0,
                half=self.gpu,     # FP16 for GPU speed
                gpu_id=0 if self.gpu else None,
            )
            self.method_name = f"Real-ESRGAN ({model_name})"
            return True

        except ImportError:
            log.info("Real-ESRGAN not installed, trying fallback...")
            return False
        except Exception as e:
            log.warning(f"Real-ESRGAN init failed: {e}")
            return False

    def _init_opencv_dnn(self) -> bool:
        """Try to initialize OpenCV DNN Super Resolution (EDSR)."""
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()

            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(exist_ok=True)

            # Try EDSR first (best quality among OpenCV DNN models)
            model_name = f"EDSR_x{self.scale}.pb"
            model_path = model_dir / model_name

            if not model_path.exists():
                log.info(f" Downloading {model_name} model...")
                try:
                    import urllib.request
                    url = (f"https://raw.githubusercontent.com/Saafke/"
                           f"EDSR_Tensorflow/master/models/{model_name}")
                    urllib.request.urlretrieve(url, str(model_path))
                    log.info(f"Model downloaded to {model_path}")
                except Exception as e:
                    log.warning(f"Auto-download failed: {e}")
                    return False

            sr.readModel(str(model_path))
            sr.setModel("edsr", self.scale)
            self._upscaler = sr
            self.method_name = f"OpenCV DNN (EDSR x{self.scale})"
            return True

        except AttributeError:
            log.info("OpenCV contrib (dnn_superres) not available, trying fallback...")
            return False
        except Exception as e:
            log.warning(f"OpenCV DNN init failed: {e}")
            return False

    def _init_pillow(self) -> bool:
        """Fallback: PIL Lanczos resize + Unsharp Mask for sharpening."""
        if Image is None:
            return False
        self._upscaler = "pillow"
        self.method_name = f"PIL Lanczos + Unsharp Mask (x{self.scale})"
        log.warning(" Using PIL fallback — quality will be lower than AI upscaling.")
        log.warning("   Install Real-ESRGAN for best results:")
        log.warning("   pip install realesrgan basicsr")
        return True

    def upscale(self, image: np.ndarray) -> np.ndarray:
        """
        Upscale an image using the best available method.
        
        Args:
            image: BGR numpy array (from cv2.imread)
        Returns:
            Upscaled BGR numpy array
        """
        if "Real-ESRGAN" in self.method_name:
            return self._upscale_realesrgan(image)
        elif "OpenCV DNN" in self.method_name:
            return self._upscale_opencv_dnn(image)
        else:
            return self._upscale_pillow(image)

    def _upscale_realesrgan(self, image: np.ndarray) -> np.ndarray:
        """Upscale using Real-ESRGAN."""
        output, _ = self._upscaler.enhance(image, outscale=self.scale)
        return output

    def _upscale_opencv_dnn(self, image: np.ndarray) -> np.ndarray:
        """Upscale using OpenCV DNN Super Resolution."""
        # DNN super-res can be memory-hungry on large images; process in tiles
        h, w = image.shape[:2]
        max_dim = 800  # Max dimension before tiling

        if max(h, w) <= max_dim:
            return self._upscaler.upsample(image)

        # Tile-based processing for large images
        tile_size = max_dim
        overlap = 16
        result = np.zeros((h * self.scale, w * self.scale, 3), dtype=np.uint8)

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tile = image[y:y_end, x:x_end]

                try:
                    upscaled_tile = self._upscaler.upsample(tile)
                except Exception:
                    # Fallback for tiles that fail
                    upscaled_tile = cv2.resize(
                        tile,
                        (tile.shape[1] * self.scale, tile.shape[0] * self.scale),
                        interpolation=cv2.INTER_LANCZOS4
                    )

                ry = y * self.scale
                rx = x * self.scale
                rh, rw = upscaled_tile.shape[:2]
                result[ry:ry + rh, rx:rx + rw] = upscaled_tile

        return result

    def _upscale_pillow(self, image: np.ndarray) -> np.ndarray:
        """Fallback upscale using PIL with aggressive sharpening."""
        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Upscale with Lanczos (best interpolation filter)
        new_size = (pil_img.width * self.scale, pil_img.height * self.scale)
        pil_img = pil_img.resize(new_size, Image.LANCZOS)

        # Aggressive unsharp mask to combat softness
        # (radius=2, percent=150, threshold=3) — adds sharpness without noise
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(
            radius=2, percent=180, threshold=3
        ))

        # Second pass with lighter sharpening for micro-detail
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(
            radius=0.5, percent=80, threshold=2
        ))

        # Convert back to BGR numpy
        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result


# Redis caching

class RedisCache:
    """
    Optional Redis cache for processed images.

    Stores final encoded image bytes (WebP/JPG/PNG) keyed by
    MD5(original file) + processing parameters.
    If Redis is not installed or the server is unreachable, the cache
    silently disables itself so the pipeline keeps working.
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, ttl: int = 86400 * 7):
        self._client = None
        self._available = False
        self._ttl = ttl

        if redis_lib is None:
            log.warning(" Библиотека redis не установлена "
                        "(pip install redis). Кэширование отключено.")
            return
        try:
            self._client = redis_lib.Redis(
                host=host, port=port, db=db,
                socket_connect_timeout=3,
            )
            self._client.ping()
            self._available = True
            log.info("Redis-кэш подключён (%s:%s, db=%s)", host, port, db)
        except Exception as exc:
            log.warning(" Не удалось подключиться к Redis: %s. "
                        "Кэширование отключено.", exc)

    #  public helpers 

    @property
    def available(self) -> bool:
        return self._available

    def make_key(self, file_path: str, scale: int,
                 output_format: str, skip_watermark: bool) -> str:
        """Build a unique cache key from file content hash + params."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        params = f"scale={scale}|fmt={output_format}|skipwm={skip_watermark}"
        hasher.update(params.encode())
        return f"imgproc:{hasher.hexdigest()}"

    def get(self, key: str):
        """Return cached bytes or *None*."""
        if not self._available:
            return None
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, data: bytes) -> None:
        """Store image bytes with TTL."""
        if not self._available:
            return
        try:
            self._client.setex(key, self._ttl, data)
        except Exception as exc:
            log.warning(" Redis set failed: %s", exc)




class ImageProcessingPipeline:
    """
    Orchestrates the full pipeline:
      ZIP → Extract → Remove Watermarks → AI Upscale → Save → Cleanup
    """

    def __init__(self, zip_path: str, output_dir: str,
                 mask_path: Optional[str] = None,
                 scale: int = 4,
                 output_format: str = "png",
                 jpg_quality: int = 100,
                 use_gpu: bool = True,
                 skip_watermark: bool = False,
                 skip_upscale: bool = False,
                 inpaint_radius: int = 7,
                 inpaint_method: str = "telea",
                 use_redis: bool = False,
                 num_workers: int = 0):
        """
        Args:
            zip_path: Path to the input .zip archive.
            output_dir: Directory to save processed images.
            mask_path: Optional path to watermark mask image.
            scale: Upscale factor (2 or 4).
            output_format: "png" (lossless) or "jpg" (lossy but smaller).
            jpg_quality: JPEG quality if output_format is "jpg" (1-100).
            use_gpu: Attempt GPU acceleration.
            skip_watermark: Skip watermark removal step.
            skip_upscale: Skip upscaling step.
            inpaint_radius: Radius for inpainting algorithm.
            inpaint_method: "telea" or "ns".
            use_redis: Enable Redis caching for processed images.
            num_workers: Number of parallel workers (0 = auto).
        """
        self.zip_path = zip_path
        self.output_dir = Path(output_dir)
        self.mask_path = mask_path
        self.scale = scale
        self.output_format = output_format.lower()
        self.jpg_quality = jpg_quality
        self.use_gpu = use_gpu
        self.skip_watermark = skip_watermark
        self.skip_upscale = skip_upscale
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = inpaint_method
        self.use_redis = use_redis
        self.num_workers = num_workers

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    #  helpers 

    def _resolve_workers(self) -> int:
        """Return the effective worker count."""
        if self.num_workers > 0:
            return self.num_workers
        # Auto: Use all available CPU threads as requested
        return os.cpu_count() or 6

    def _process_single_image(
        self,
        img_path: str,
        watermark_remover,
        upscaler,
        upscale_lock: threading.Semaphore,
        cache,
    ) -> Tuple[bool, bool, str]:
        """
        Process one image (thread-safe).

        Returns:
            (success, from_cache, log_message)
        """
        filename = os.path.basename(img_path)

        if not is_valid_img(img_path):
            return False, False, f" Skipping corrupt file: {filename}"

        #  Determine output filename 
        stem = Path(filename).stem
        if self.output_format == "png":
            out_name = f"{stem}.png"
        elif self.output_format == "webp":
            out_name = f"{stem}.webp"
        else:
            out_name = f"{stem}.jpg"
        out_path = self.output_dir / out_name

        #  Check Redis cache 
        cache_key = None
        if cache is not None:
            cache_key = cache.make_key(
                img_path, self.scale,
                self.output_format, self.skip_watermark,
            )
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                with open(str(out_path), "wb") as fh:
                    fh.write(cached_data)
                return True, True, f"  {filename} → из кэша → {out_name}"

        #  Load image 
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        msg_parts = [f"  {filename} ({w}x{h})"]

        # Remove watermark (CPU-bound — runs in parallel)
        if watermark_remover is not None:
            image = watermark_remover.remove(image)

        # Upscale (GPU-lock — serialised across threads)
        if upscaler is not None:
            with upscale_lock:
                image = upscaler.upscale(image)

        #  Encode output bytes 
        if self.output_format == "png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            _, buf = cv2.imencode(".png", image, encode_params)
        elif self.output_format == "webp":
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, self.jpg_quality]
            _, buf = cv2.imencode(".webp", image, encode_params)
        else:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
            _, buf = cv2.imencode(".jpg", image, encode_params)

        img_bytes = buf.tobytes()

        #  Save to disk 
        with open(str(out_path), "wb") as fh:
            fh.write(img_bytes)

        #  Store in Redis cache 
        if cache is not None and cache_key is not None:
            cache.set(cache_key, img_bytes)

        new_h, new_w = image.shape[:2]
        msg_parts.append(f"    → {out_name} ({new_w}x{new_h})")
        return True, False, "\n".join(msg_parts)

    def run(self):
        """Execute the full pipeline."""
        start_time = time.time()
        workers = self._resolve_workers()

        log.info("=" * 60)
        log.info("  PREMIUM IMAGE PROCESSOR — Pipeline Start")
        log.info("=" * 60)
        log.info(f"  Input:  {self.zip_path}")
        log.info(f"  Output: {self.output_dir}")
        log.info(f"  Scale:  {self.scale}x")
        log.info(f"  Format: {self.output_format.upper()}")
        log.info(f"  Workers: {workers}")
        if self.skip_watermark:
            log.info("  ⏩ Watermark removal: SKIPPED")
        if self.skip_upscale:
            log.info("  ⏩ Upscaling: SKIPPED")
        log.info("=" * 60)

        #  Create temp directory 
        temp_dir = tempfile.mkdtemp(prefix="imgproc_")
        log.info(f"Temp directory: {temp_dir}")

        #  Init Redis cache (if requested) 
        cache = None
        if self.use_redis:
            cache = RedisCache()
            if not cache.available:
                cache = None  # fall back to normal processing

        try:
            #  Step 1: Extract 
            log.info("\nSTEP 1/3: Extracting images from ZIP")
            extractor = ZipExtractor(self.zip_path, temp_dir)
            image_paths = extractor.extract()

            if not image_paths:
                log.error("No valid images found in the archive!")
                return

            #  Step 2: Init watermark remover 
            watermark_remover = None
            if not self.skip_watermark:
                log.info("\nSTEP 2/3: Watermark Removal")
                watermark_remover = WatermarkRemover(
                    mask_path=self.mask_path,
                    inpaint_radius=self.inpaint_radius,
                    method=self.inpaint_method,
                )

            #  Step 3: Init upscaler 
            upscaler = None
            if not self.skip_upscale:
                log.info("\nSTEP 3/3: AI Upscaling")
                upscaler = AIUpscaler(scale=self.scale, gpu=self.use_gpu)

            #  Process images in parallel 
            # Use Semaphore(3) to allow multiple CPU threads to push work to GPU
            # and achieve ~100% GPU utilization instead of strictly 1.
            upscale_lock = threading.Semaphore(3)
            log.info(f"\nProcessing {len(image_paths)} image(s) "
                     f"with {workers} worker(s)...\n")

            successful = 0
            failed = 0
            cached_hits = 0

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
            ) as executor:
                future_to_path = {
                    executor.submit(
                        self._process_single_image,
                        img_path, watermark_remover, upscaler,
                        upscale_lock, cache,
                    ): img_path
                    for img_path in image_paths
                }
                for future in concurrent.futures.as_completed(future_to_path):
                    img_path = future_to_path[future]
                    filename = os.path.basename(img_path)
                    try:
                        ok, from_cache, msg = future.result()
                        if ok:
                            successful += 1
                            if from_cache:
                                cached_hits += 1
                        else:
                            failed += 1
                        log.info(msg)
                    except Exception as exc:
                        log.error(f"Failed to process {filename}: {exc}")
                        failed += 1

        finally:
            #  Cleanup temp directory 
            try:
                shutil.rmtree(temp_dir)
                log.info(f"\n Cleaned up temp directory")
            except Exception as e:
                log.warning(f"Could not clean temp dir: {e}")

        #  Summary 
        elapsed = time.time() - start_time
        log.info("Finished in %.1fs", elapsed)




def main():
    parser = argparse.ArgumentParser(
        description="Premium Image Processor — Remove watermarks & AI-upscale images from a ZIP archive.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES


Basic usage (auto-detect watermarks, 4x upscale, PNG output):
python image_processor.py images.zip

With a watermark mask, 2x upscale, JPG output:
python image_processor.py images.zip
--mask watermark_mask.png --scale 2 --format jpg

Skip watermark removal (upscale only):
python image_processor.py images.zip --skip-watermark

Skip upscaling (watermark removal only):
python image_processor.py images.zip --skip-upscale

Custom output directory and CPU-only mode:
python image_processor.py images.zip
--output ./processed --no-gpu

        """
    )

    parser.add_argument("zip_path",
                        help="Path to the .zip archive with images")
    parser.add_argument("--output", "-o", default="./output_processed",
                        help="Output directory (default: ./output_processed)")
    parser.add_argument("--mask", "-m", default=None,
                        help="Path to watermark mask image (white = watermark area)")
    parser.add_argument("--scale", "-s", type=int, default=2, choices=[2, 4],
                        help="Upscale factor: 2 or 4 (default: 2)")
    parser.add_argument("--format", "-f", default="webp", choices=["png", "jpg", "webp"],
                        help="Output format (default: webp)")
    parser.add_argument("--jpg-quality", type=int, default=100,
                        help="JPEG quality 1-100 (default: 100)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Force CPU mode (no CUDA)")
    parser.add_argument("--skip-watermark", action="store_true",
                        help="Skip watermark removal step")
    parser.add_argument("--skip-upscale", action="store_true",
                        help="Skip upscaling step")
    parser.add_argument("--inpaint-radius", type=int, default=7,
                        help="Inpainting radius in pixels (default: 7)")
    parser.add_argument("--inpaint-method", default="telea",
                        choices=["telea", "ns"],
                        help="Inpainting method (default: telea)")

    args = parser.parse_args()

    pipeline = ImageProcessingPipeline(
        zip_path=args.zip_path,
        output_dir=args.output,
        mask_path=args.mask,
        scale=args.scale,
        output_format=args.format,
        jpg_quality=args.jpg_quality,
        use_gpu=not args.no_gpu,
        skip_watermark=args.skip_watermark,
        skip_upscale=args.skip_upscale,
        inpaint_radius=args.inpaint_radius,
        inpaint_method=args.inpaint_method,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
