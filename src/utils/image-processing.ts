interface ProcessedImage {
  element: HTMLImageElement;
  width: number;
  height: number;
  aspectRatio: number;
  dataUrl: string;
}

interface ProcessingOptions {
  maxSize?: number;
  minSize?: number;
  quality?: number;
  format?: 'image/jpeg' | 'image/png' | 'image/webp';
  task?: 'classification' | 'detection';
  enhanceContrast?: boolean;
  denoise?: boolean;
  sharpen?: boolean;
  provider?: 'huggingface';
}

const DEFAULT_OPTIONS = {
  maxSize: 1024,
  minSize: 224,
  quality: 0.9,
  format: 'image/jpeg' as const,
  task: 'classification' as const,
  enhanceContrast: true,
  denoise: true,
  sharpen: true
};

// Helper functions for image processing
function convertToGrayscale(data: Uint8ClampedArray): Uint8ClampedArray {
  const grayscale = new Uint8ClampedArray(data.length / 4);
  for (let i = 0; i < data.length; i += 4) {
    grayscale[i / 4] = Math.round(
      0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
    );
  }
  return grayscale;
}

function applyMedianFilter(
  grayscale: Uint8ClampedArray,
  width: number,
  height: number,
  radius: number = 1,
  isDetection: boolean = false
): Uint8ClampedArray {
  // Use smaller radius for detection to preserve edges
  const filterRadius = isDetection ? 1 : radius;
  const temp = new Uint8ClampedArray(grayscale);
  const result = new Uint8ClampedArray(grayscale);

  for (let y = filterRadius; y < height - filterRadius; y++) {
    for (let x = filterRadius; x < width - filterRadius; x++) {
      const values = [];
      for (let dy = -filterRadius; dy <= filterRadius; dy++) {
        for (let dx = -filterRadius; dx <= filterRadius; dx++) {
          values.push(temp[(y + dy) * width + (x + dx)]);
        }
      }
      values.sort((a, b) => a - b);
      result[y * width + x] = values[Math.floor(values.length / 2)];
    }
  }
  return result;
}

function applySharpeningFilter(
  grayscale: Uint8ClampedArray,
  width: number,
  height: number,
  isDetection: boolean = false
): Uint8ClampedArray {
  // Use different kernel strengths based on task
  const kernel = isDetection ? [
    [0, -0.5, 0],
    [-0.5, 3, -0.5],
    [0, -0.5, 0]
  ] : [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
  ];
  
  const temp = new Uint8ClampedArray(grayscale);
  const result = new Uint8ClampedArray(grayscale);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sum = 0;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          sum += temp[(y + ky - 1) * width + (x + kx - 1)] * kernel[ky][kx];
        }
      }
      result[y * width + x] = Math.max(0, Math.min(255, sum));
    }
  }
  return result;
}

// Calculate block statistics for adaptive contrast
function calculateBlockStats(
  grayscale: Uint8ClampedArray,
  width: number,
  bx: number,
  by: number,
  blockWidth: number,
  blockHeight: number
): { sum: number; min: number; max: number } {
  let sum = 0, min = 255, max = 0;
  
  for (let y = 0; y < blockHeight; y++) {
    for (let x = 0; x < blockWidth; x++) {
      const value = grayscale[(by + y) * width + (bx + x)];
      sum += value;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  
  return { sum, min, max };
}

// Helper function to get error message
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (error instanceof Event) {
    return 'Image loading failed';
  }
  return String(error);
}

interface BlockParameters {
  grayscale: Uint8ClampedArray;
  result: Uint8ClampedArray;
  width: number;
  position: {
    x: number;
    y: number;
    blockWidth: number;
    blockHeight: number;
  };
  contrast: {
    mean: number;
    scale: number;
  };
}

// Apply contrast enhancement to a single block
function enhanceBlockContrast(params: BlockParameters): void {
  const { grayscale, result, width, position, contrast } = params;
  const { x: bx, y: by, blockWidth, blockHeight } = position;
  const { mean, scale } = contrast;

  for (let y = 0; y < blockHeight; y++) {
    for (let x = 0; x < blockWidth; x++) {
      const idx = (by + y) * width + (bx + x);
      const value = grayscale[idx];
      result[idx] = Math.max(0, Math.min(255,
        mean + scale * (value - mean)
      ));
    }
  }
}

function applyAdaptiveContrast(
  grayscale: Uint8ClampedArray,
  width: number,
  height: number,
  blockSize: number = 8,
  maxContrast: number = 2.0,
  isDetection: boolean = false
): Uint8ClampedArray {
  // Use more conservative contrast enhancement for detection
  const contrastLimit = isDetection ? 1.5 : maxContrast;
  const result = new Uint8ClampedArray(grayscale);

  for (let by = 0; by < height; by += blockSize) {
    for (let bx = 0; bx < width; bx += blockSize) {
      const blockWidth = Math.min(blockSize, width - bx);
      const blockHeight = Math.min(blockSize, height - by);

      const { sum, min, max } = calculateBlockStats(
        grayscale,
        width,
        bx,
        by,
        blockWidth,
        blockHeight
      );

      const mean = sum / (blockWidth * blockHeight);
      const contrast = (max - min) / 255;
      const scale = Math.min(contrastLimit, 1 / contrast);

      enhanceBlockContrast({
        grayscale,
        result,
        width,
        position: {
          x: bx,
          y: by,
          blockWidth,
          blockHeight
        },
        contrast: {
          mean,
          scale
        }
      });
    }
  }
  
  return result;
}

function applyEdgeDetection(
  grayscale: Uint8ClampedArray,
  width: number,
  height: number
): Uint8ClampedArray {
  const sobelX = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
  ];
  const sobelY = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
  ];
  const temp = new Uint8ClampedArray(grayscale);
  const result = new Uint8ClampedArray(grayscale);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const val = temp[(y + ky - 1) * width + (x + kx - 1)];
          gx += val * sobelX[ky][kx];
          gy += val * sobelY[ky][kx];
        }
      }
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      result[y * width + x] = Math.min(255, magnitude);
    }
  }
  return result;
}

function applyEnhancedGrayscale(
  data: Uint8ClampedArray,
  grayscale: Uint8ClampedArray
): void {
  for (let i = 0; i < data.length; i += 4) {
    const enhanced = grayscale[i / 4];
    const original = (data[i] + data[i + 1] + data[i + 2]) / 3;
    const ratio = enhanced / (original || 1);
    data[i] = Math.max(0, Math.min(255, data[i] * ratio));
    data[i + 1] = Math.max(0, Math.min(255, data[i + 1] * ratio));
    data[i + 2] = Math.max(0, Math.min(255, data[i + 2] * ratio));
  }
}

/**
 * Applies image enhancement filters to improve ML model performance
 */
function applyImageFilters(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  options: Required<ProcessingOptions>
): void {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const isDetection = options.task === 'detection';

  if (isDetection) {
    // For detection: minimal preprocessing or no preprocessing at all
    // Most object detection models work better with raw images
    return;
  } else {
    // For classification: full enhancement pipeline
    let grayscale = convertToGrayscale(data);

    if (options.denoise) {
      grayscale = applyMedianFilter(grayscale, width, height, 1, false);
    }
    if (options.sharpen) {
      grayscale = applySharpeningFilter(grayscale, width, height, false);
    }
    if (options.enhanceContrast) {
      grayscale = applyAdaptiveContrast(grayscale, width, height, 8, 2.0, false);
    }

    // Apply enhanced grayscale back to color image
    applyEnhancedGrayscale(data, grayscale);
    ctx.putImageData(imageData, 0, 0);
  }
}

/**
 * Processes an image for machine learning tasks
 * - Resizes the image while maintaining aspect ratio
 * - Ensures minimum dimensions for ML models
 * - Optimizes quality and file size
 * - Converts to consistent format
 * - Applies task-specific enhancements
 */
export async function preprocessImage(
  image: HTMLImageElement,
  options: ProcessingOptions = {}
): Promise<ProcessedImage> {
  return new Promise((resolve, reject) => {
    try {
      const processOptions = { ...DEFAULT_OPTIONS, ...options } as Required<ProcessingOptions>;
      const isDetection = processOptions.task === 'detection';
      const isHuggingFace = processOptions.provider === 'huggingface';

      // Create canvas for image processing
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        throw new Error('Could not get canvas context');
      }

      // Get original dimensions
      let width = image.naturalWidth;
      let height = image.naturalHeight;
      const aspectRatio = width / height;

      // Check minimum dimensions
      if (width < processOptions.minSize || height < processOptions.minSize) {
        throw new Error(`Image dimensions must be at least ${processOptions.minSize}x${processOptions.minSize} pixels`);
      }

      // For Hugging Face models, use their preferred input size
      const maxSize = isHuggingFace ? 800 : (isDetection ? Math.min(640, processOptions.maxSize) : processOptions.maxSize);

      // Scale down if image is too large
      if (width > maxSize || height > maxSize) {
        if (width > height) {
          height = Math.round((height * maxSize) / width);
          width = maxSize;
        } else {
          width = Math.round((width * maxSize) / height);
          height = maxSize;
        }
      }

      // Set canvas dimensions
      canvas.width = width;
      canvas.height = height;

      // Apply image smoothing for better quality
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // Draw initial image
      ctx.drawImage(image, 0, 0, width, height);

      // Apply enhancement filters (will be skipped for detection)
      applyImageFilters(ctx, width, height, processOptions);

      // Convert to desired format with appropriate quality
      const quality = isHuggingFace ? 1.0 : (isDetection ? 1.0 : processOptions.quality); // Use max quality for Hugging Face
      const format = isHuggingFace ? 'image/png' : processOptions.format; // Use PNG for Hugging Face
      const dataUrl = canvas.toDataURL(format, quality);

      // Create new image with processed data
      const processedImage = new Image();
      processedImage.crossOrigin = 'anonymous';
      processedImage.src = dataUrl;

      processedImage.onload = () => resolve({
        element: processedImage,
        width,
        height,
        aspectRatio,
        dataUrl
      });

      processedImage.onerror = (error) => {
        const errorMessage = getErrorMessage(error);
        reject(new Error(`Failed to load processed image: ${errorMessage}`));
      };
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      reject(new Error(`Image processing failed: ${errorMessage}`));
    }
  });
}

/**
 * Validates an image file before processing
 * - Checks file type
 * - Verifies dimensions
 * - Ensures file size is within limits
 */
export function validateImageFile(
  file: File,
  maxSizeMB: number = 5
): Promise<string | null> {
  return new Promise((resolve, reject) => {
    // Check file size
    if (file.size > maxSizeMB * 1024 * 1024) {
      reject(new Error(`File is too large. Maximum size is ${maxSizeMB}MB`));
      return;
    }

    // Check file type
    if (!file.type.startsWith('image/')) {
      reject(new Error('Invalid file type. Please upload an image file'));
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Check minimum dimensions
        if (img.naturalWidth < 224 || img.naturalHeight < 224) {
          reject(new Error('Image dimensions must be at least 224x224 pixels'));
          return;
        }
        resolve(e.target?.result as string);
      };
      img.onerror = () => {
        reject(new Error('Failed to load and validate image'));
      };
      img.src = e.target?.result as string;
    };
    reader.onerror = () => {
      reject(new Error('Error reading image file'));
    };
    reader.readAsDataURL(file);
  });
}

/**
 * Creates a canvas element for drawing ML model results
 */
export function createResultCanvas(
  image: HTMLImageElement,
  maxWidth: number = 1024,
  maxHeight: number = 1024
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    throw new Error('Could not get canvas context');
  }

  // Calculate dimensions maintaining aspect ratio
  let { width, height } = image;
  const aspectRatio = width / height;

  if (width > maxWidth) {
    width = maxWidth;
    height = Math.round(width / aspectRatio);
  }

  if (height > maxHeight) {
    height = maxHeight;
    width = Math.round(height * aspectRatio);
  }

  canvas.width = width;
  canvas.height = height;

  // Store original image for redrawing
  canvas.dataset.originalImage = image.src;

  // Draw initial image
  ctx.drawImage(image, 0, 0, width, height);

  return canvas;
} 