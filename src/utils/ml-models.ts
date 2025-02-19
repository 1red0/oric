import * as mobilenet from '@tensorflow-models/mobilenet';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs';
import { preprocessImage } from './image-processing';
import { pipeline } from '@huggingface/transformers';

let mobilenetModel: mobilenet.MobileNet | null = null;
let cocoSsdModel: cocoSsd.ObjectDetection | null = null;
let mobilenetLoadingPromise: Promise<mobilenet.MobileNet> | null = null;
let cocoSsdLoadingPromise: Promise<cocoSsd.ObjectDetection> | null = null;
let backendInitialized = false;
let isModelLoading = false;

let resnet50Pipeline: any = null;
let resnet101Pipeline: any = null;
let mobilevitPipeline: any = null;
let detrResnet50Pipeline: any = null;
let detrResnet101Pipeline: any = null;
let yolosBasePipeline: any = null;

export const models = {
  classification: [
    {
      id: 'mobilenet',
      name: 'MobileNet',
      provider: 'tensorflow' as const,
    },
    {
      id: 'Xenova/resnet-50',
      name: 'ResNet-50',
      provider: 'huggingface' as const,
    },
    {
      id: 'Xenova/resnet-101',
      name: 'ResNet-101',
      provider: 'huggingface' as const,
    },
    {
      id: 'Xenova/mobilevit-small',
      name: 'MobileViT',
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
      id: 'Xenova/detr-resnet-50',
      name: 'DETR ResNet-50',
      provider: 'huggingface' as const,
    },
    {
      id: 'Xenova/detr-resnet-101',
      name: 'DETR ResNet-101',
      provider: 'huggingface' as const,
    },
    {
      id: 'Xenova/yolos-base',
      name: 'YOLOS Base',
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

const pipelineMap = {
  'Xenova/resnet-50': () => resnet50Pipeline,
  'Xenova/resnet-101': () => resnet101Pipeline,
  'Xenova/mobilevit-small': () => mobilevitPipeline,
  'Xenova/detr-resnet-50': () => detrResnet50Pipeline,
  'Xenova/detr-resnet-101': () => detrResnet101Pipeline,
  'Xenova/yolos-base': () => yolosBasePipeline,
} as const;

async function loadPipeline(task: 'image-classification' | 'object-detection', modelId: string) {
  const getPipeline = pipelineMap[modelId as keyof typeof pipelineMap];
  if (!getPipeline) {
    throw new Error(`Unsupported model: ${modelId}`);
  }

  let pipelineRef = getPipeline();
  if (!pipelineRef) {
    pipelineRef = await pipeline(task, modelId);
    switch (modelId) {
      case 'Xenova/resnet-50': resnet50Pipeline = pipelineRef; break;
      case 'Xenova/resnet-101': resnet101Pipeline = pipelineRef; break;
      case 'Xenova/mobilevit-small': mobilevitPipeline = pipelineRef; break;
      case 'Xenova/detr-resnet-50': detrResnet50Pipeline = pipelineRef; break;
      case 'Xenova/detr-resnet-101': detrResnet101Pipeline = pipelineRef; break;
      case 'Xenova/yolos-base': yolosBasePipeline = pipelineRef; break;
    }
  }
  return pipelineRef;
}

export async function classifyImage(modelId: string, imageData: string): Promise<Array<{ label: string; score: number }>> {
  if (isModelLoading) {
    throw new Error('Another recognition is in progress. Please wait.');
  }

  try {
    isModelLoading = true;
    const img = await createImageFromDataUrl(imageData);
    const processedImage = await preprocessImage(img, {
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
      const results = await model.classify(processedImage.element);
      return results.map(result => ({
        label: result.className,
        score: result.probability
      }));
    } else {
      const pipe = await loadPipeline('image-classification', modelId);
      const results = await pipe(processedImage.dataUrl);
      return results.map((result: { label: string; score: number }) => ({
        label: result.label,
        score: result.score
      }));
    }
  } catch (error) {
    throw new Error(`Classification failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
  } finally {
    isModelLoading = false;
  }
}

export async function detectObjects(modelId: string, imageData: string): Promise<Array<{ bbox: number[]; class: string; score: number }>> {
  if (isModelLoading) {
    throw new Error('Another recognition is in progress. Please wait.');
  }

  try {
    isModelLoading = true;
    const img = await createImageFromDataUrl(imageData);
    const processedImage = await preprocessImage(img, {
      maxSize: 1024,
      minSize: 224,
      quality: 0.9,
      task: 'detection',
      enhanceContrast: true,
      denoise: true,
      sharpen: true
    });

    if (modelId === 'coco-ssd') {
      const model = await loadCocoSsd();
      const results = await model.detect(processedImage.element);
      return results
        .filter(result => result.score > 0.35)
        .map(result => ({
          bbox: [result.bbox[0], result.bbox[1], result.bbox[2], result.bbox[3]],
          class: result.class,
          score: result.score
        }));
    } else {
      const pipe = await loadPipeline('object-detection', modelId);
      const results = await pipe(processedImage.dataUrl);
      return results
        .filter((det: any) => det.score > 0.35)
        .map((result: { box: { xmin: number; ymin: number; xmax: number; ymax: number }; label: string; score: number }) => ({
          bbox: [result.box.xmin, result.box.ymin, result.box.xmax, result.box.ymax],
          class: result.label,
          score: result.score
        }));
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    throw new Error(`Object recognition failed: ${errorMessage}`);
  } finally {
    isModelLoading = false;
  }
}

async function createImageFromDataUrl(dataUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
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