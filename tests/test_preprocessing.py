import cv2
import os
import sys


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Add project root to Python path
sys.path.insert(
    0,
    PROJECT_ROOT
)


# ==========================================================
# IMPORT PREPROCESSING MODULES
# ==========================================================

from src.preprocessing.quality import ImageQualityAnalyzer
from src.preprocessing.enhancement import ImageEnhancer


# ==========================================================
# IMAGE PATH
# ==========================================================

IMAGE_PATH = os.path.join(
    PROJECT_ROOT,
    "test_images",
    "sensitive",
    "test4.jpg"
)


# ==========================================================
# LOAD IMAGE
# ==========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


print("=" * 60)
print("VISIONGUARD - IMAGE PREPROCESSING TEST")
print("=" * 60)


# ==========================================================
# QUALITY ANALYSIS
# ==========================================================

quality_analyzer = ImageQualityAnalyzer()

quality_result = quality_analyzer.analyze(image)


print("\nQUALITY RESULT")
print("-" * 60)

for key, value in quality_result.items():
    print(f"{key:20s}: {value}")


# ==========================================================
# ENHANCEMENT
# ==========================================================

enhancer = ImageEnhancer()

enhanced_image = enhancer.enhance(
    image,
    quality_result
)


# ==========================================================
# SAVE RESULT
# ==========================================================

output_directory = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    output_directory,
    exist_ok=True
)

output_path = os.path.join(
    output_directory,
    "enhanced_test.jpg"
)

success = cv2.imwrite(
    output_path,
    enhanced_image
)


# ==========================================================
# RESULT
# ==========================================================

print("\nENHANCEMENT RESULT")
print("-" * 60)

if success:
    print(f"Enhanced image saved to:")
    print(output_path)
else:
    print("Failed to save enhanced image.")

print("=" * 60)