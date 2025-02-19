import * as mobilenet from '@tensorflow-models/mobilenet';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { HfInference } from '@huggingface/inference';
import * as tf from '@tensorflow/tfjs';
import { preprocessImage } from './image-processing';

let mobilenetModel: mobilenet.MobileNet | null = null;
let cocoSsdModel: cocoSsd.ObjectDetection | null = null;
let mobilenetLoadingPromise: Promise<mobilenet.MobileNet> | null = null;
let cocoSsdLoadingPromise: Promise<cocoSsd.ObjectDetection> | null = null;
let backendInitialized = false;
let isModelLoading = false;

export const models = {
  classification: [
    {
      id: 'mobilenet',
      name: 'MobileNet',
      provider: 'tensorflow' as const,
    },
    {
      id: 'microsoft/resnet-50',
      name: 'ResNet-50 ',
      provider: 'huggingface' as const,
    },
  ],
  recognition: [
    {
      id: 'coco-ssd',
      name: 'COCO-SSD',
      provider: 'tensorflow' as const,
    },
    {
      id: 'facebook/detr-resnet-50',
      name: 'DETR ',
      provider: 'huggingface' as const,
    },
  ],
};

async function initializeBackend() {
  if (!backendInitialized) {
    try {
      // Try to use WebGL backend first
      await tf.setBackend('webgl');
      await tf.ready();
      console.log('Using WebGL backend');
    } catch (e) {
      // Fallback to CPU backend if WebGL is not available
      console.log('WebGL backend not available. Falling back to CPU backend. ' + e);
      try {
        await tf.setBackend('cpu');
        await tf.ready();
        console.log('Using CPU backend');
      } catch (e) {
        throw new Error('Failed to initialize TensorFlow.js backend. Please make sure your browser supports WebGL. ' + e);
      }
    }
    backendInitialized = true;
  }
}

async function loadMobilenet(): Promise<mobilenet.MobileNet> {
  await initializeBackend();
  
  if (!mobilenetModel && !mobilenetLoadingPromise) {
    mobilenetLoadingPromise = mobilenet.load();
    try {
      mobilenetModel = await mobilenetLoadingPromise;
    } finally {
      mobilenetLoadingPromise = null;
    }
  } else if (mobilenetLoadingPromise) {
    try {
      mobilenetModel = await mobilenetLoadingPromise;
    } finally {
      mobilenetLoadingPromise = null;
    }
  }
  if (!mobilenetModel) throw new Error('Failed to load MobileNet model');
  return mobilenetModel;
}

async function loadCocoSsd(): Promise<cocoSsd.ObjectDetection> {
  await initializeBackend();
  
  if (!cocoSsdModel && !cocoSsdLoadingPromise) {
    cocoSsdLoadingPromise = cocoSsd.load();
    try {
      cocoSsdModel = await cocoSsdLoadingPromise;
    } finally {
      cocoSsdLoadingPromise = null;
    }
  } else if (cocoSsdLoadingPromise) {
    try {
      cocoSsdModel = await cocoSsdLoadingPromise;
    } finally {
      cocoSsdLoadingPromise = null;
    }
  }
  if (!cocoSsdModel) throw new Error('Failed to load COCO-SSD model');
  return cocoSsdModel;
}

export async function classifyImage(
  modelId: string,
  image: HTMLImageElement,
  hfToken?: string
) {
  try {
    // Pre-process image with classification-specific enhancements
    const processedImage = await preprocessImage(image, {
      maxSize: 1024,
      minSize: 224,
      quality: 0.9,
      task: 'classification',
      enhanceContrast: true,
      denoise: true,
      sharpen: true
    });

    if (modelId === 'mobilenet') {
      const model = await loadMobilenet();
      return await model.classify(processedImage.element);
    } else if (modelId === 'microsoft/resnet-50') {
      if (!hfToken) throw new Error('Hugging Face token is required');
      const hf = new HfInference(hfToken);
      const response = await hf.imageClassification({
        model: modelId,
        data: processedImage.dataUrl,
      });
      return response.map(({ label, score }) => ({
        className: label,
        probability: score,
      }));
    }
    throw new Error(`Unsupported model: ${modelId}`);
  } catch (error) {
    throw new Error(`Classification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

async function detectWithCocoSsd(processedImage: { element: HTMLImageElement }) {
  const model = await loadCocoSsd();
  const results = await model.detect(processedImage.element);
  return results.filter(result => result.score > 0.35);
}

async function detectWithHuggingFace(
  modelId: string,
  processedImage: { dataUrl: string },
  hfToken: string
) {
  if (!hfToken?.trim() || !hfToken.startsWith('hf_')) {
    throw new Error('Invalid Hugging Face token format. Token should start with "hf_".');
  }

  const hf = new HfInference(hfToken);
  const response = await hf.objectDetection({
    model: modelId,
    data: processedImage.dataUrl,
  });

  if (!response || !Array.isArray(response)) {
    throw new Error('Invalid response from Hugging Face API');
  }

  return response
    .filter(det => det.score > 0.35)
    .map(({ label, score, box }) => ({
      class: label,
      score,
      bbox: [box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin],
    }));
}

export async function detectObjects(
  modelId: string,
  image: HTMLImageElement,
  hfToken?: string
) {
  if (isModelLoading) {
    throw new Error('Another recognition is in progress. Please wait.');
  }

  try {
    isModelLoading = true;
    const processedImage = await preprocessImage(image, {
      maxSize: 1024,
      minSize: 224,
      quality: 0.9,
      task: 'detection',
      enhanceContrast: true,
      denoise: true,
      sharpen: true,
      provider: modelId.includes('facebook/') || modelId.includes('microsoft/') ? 'huggingface' : undefined
    });

    if (modelId === 'coco-ssd') {
      return await detectWithCocoSsd(processedImage);
    } 
    
    if (modelId === 'facebook/detr-resnet-50') {
      if (!hfToken) {
        throw new Error('Hugging Face token is missing. Please check your .env.local file.');
      }
      return await detectWithHuggingFace(modelId, processedImage, hfToken);
    }

    throw new Error(`Unsupported model: ${modelId}`);
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    throw new Error(`Object recognition failed: ${errorMessage}`);
  } finally {
    isModelLoading = false;
  }
}

export function drawRecognitions(
  canvas: HTMLCanvasElement,
  recognitions: Array<{
    bbox: number[];
    class?: string;
    label?: string;
    score: number;
  }>
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Clear previous drawings
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw the original image first
  if (ctx.canvas.dataset.originalImage) {
    const img = new Image();
    img.src = ctx.canvas.dataset.originalImage;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }

  recognitions.forEach((recognition) => {
    const [x, y, width, height] = recognition.bbox;
    const label = recognition.class ?? recognition.label ?? 'Unknown';
    const score = Math.round(recognition.score * 100);

    // Draw bounding box with dashed border
    ctx.strokeStyle = '#4B5320';
    ctx.lineWidth = 3;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(x, y, width, height);
    ctx.setLineDash([]); // Reset line dash

    // Prepare label text
    const labelText = `${label} ${score}%`;
    ctx.font = 'bold 16px Inter, Arial';
    const textMetrics = ctx.measureText(labelText);
    const textWidth = textMetrics.width;
    const textHeight = 20;
    const padding = 8;

    // Calculate label background position
    const bgX = x;
    const bgY = y > textHeight + padding * 2 ? y - textHeight - padding * 2 : y + height + padding;

    // Draw label background with gradient
    const gradient = ctx.createLinearGradient(bgX, bgY, bgX + textWidth + padding * 2, bgY);
    gradient.addColorStop(0, '#4B5320');
    gradient.addColorStop(1, '#5B6330');
    
    ctx.fillStyle = gradient;
    ctx.roundRect(bgX, bgY, textWidth + padding * 2, textHeight + padding, 4);
    ctx.fill();

    // Draw label text with shadow
    ctx.fillStyle = '#FFFFFF';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
    ctx.shadowBlur = 4;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 1;
    ctx.fillText(labelText, bgX + padding, bgY + textHeight);
    ctx.shadowColor = 'transparent';
  });
} 