'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ImageUpload from '@/components/ImageUpload';
import ModelSelector from '@/components/ModelSelector';
import { models, detectObjects } from '@/utils/ml-models';
import { ArrowPathIcon, ArrowLeftIcon, CubeTransparentIcon } from '@heroicons/react/24/outline';
import Link from 'next/link';

/**
 * Interface for recognition results
 */
interface Recognition {
  bbox: number[];
  class: string;
  score: number;
}

// Add these interfaces at the top of the file, after the Recognition interface
interface CanvasDimensions {
  width: number;
  height: number;
}

interface ScalingParameters {
  scaledWidth: number;
  scaledHeight: number;
  offsetX: number;
  offsetY: number;
}

interface Coordinates {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * ObjectRecognition component handles the recognition of objects in images using
 * various ML models (COCO-SSD from TensorFlow.js or DETR from Hugging Face).
 */
export default function ObjectRecognition() {
  // State management
  const [selectedModel, setSelectedModel] = useState(models.recognition[0]);
  const [image, setImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [recognitions, setRecognitions] = useState<Recognition[] | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasNoRecognitions, setHasNoRecognitions] = useState(false);

  /**
   * Handles image selection and resets related states
   */
  const handleImageSelect = (imageData: string | null) => {
    setImage(imageData);
    setProcessedImage(null);
    setRecognitions(null);
    setError(null);
    setHasNoRecognitions(false);
  };

  /**
   * Prepares a canvas with standardized dimensions for model input
   */
  function prepareInputCanvas(
    img: HTMLImageElement,
    modelId: string
  ): { canvas: HTMLCanvasElement; offsetX: number; offsetY: number; scale: number } {
    const standardWidth = modelId === 'coco-ssd' ? 640 : 800;
    const standardHeight = modelId === 'coco-ssd' ? 480 : 600;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Could not get input canvas context');

    canvas.width = standardWidth;
    canvas.height = standardHeight;

    const scale = Math.min(standardWidth / img.width, standardHeight / img.height);
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    const offsetX = (standardWidth - scaledWidth) / 2;
    const offsetY = (standardHeight - scaledHeight) / 2;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, standardWidth, standardHeight);

    // Draw image with high quality
    ctx.filter = 'contrast(1.2) brightness(1.1) saturate(1.2)';
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

    return { canvas, offsetX, offsetY, scale };
  }

  /**
   * Creates and prepares the output canvas for drawing results
   */
  function prepareOutputCanvas(img: HTMLImageElement): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get output canvas context');
    
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  }

  /**
   * Processes the selected image for object recognition
   */
  const processImage = async () => {
    if (!image) return;

    setIsProcessing(true);
    setError(null);
    setRecognitions(null);
    setProcessedImage(null);
    setHasNoRecognitions(false);

    try {
      // Create and load the image first
      const img = new Image();
      img.src = image;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      // Prepare input canvas with proper scaling
      const { canvas: inputCanvas, offsetX, offsetY, scale } = prepareInputCanvas(img, selectedModel.id);
      const outputCanvas = prepareOutputCanvas(img);
      const outputCtx = outputCanvas.getContext('2d');
      if (!outputCtx) throw new Error('Could not get output canvas context');

      // Use the appropriate input for each model type
      const modelInput = selectedModel.id === 'coco-ssd' ? 
        inputCanvas.toDataURL() : 
        image;

      const results = await detectObjects(
        selectedModel.id,
        modelInput
      );

      if (!results || results.length === 0) {
        setHasNoRecognitions(true);
        setError('No objects detected. Try another image or a different model.');
        return;
      }

      setRecognitions(results);

      // Draw original image on output canvas
      outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);

      // Draw each recognition
      results.forEach(recognition => {
        const canvasDimensions = {
          width: outputCanvas.width,
          height: outputCanvas.height
        };

        const scaling = {
          scaledWidth: img.width * scale,
          scaledHeight: img.height * scale,
          offsetX,
          offsetY
        };

        drawRecognitionResult(
          outputCtx,
          recognition,
          canvasDimensions,
          selectedModel.id,
          scaling
        );
      });

      setProcessedImage(outputCanvas.toDataURL('image/png'));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

  /**
   * Converts coordinates based on model type and canvas dimensions
   */
  function convertCoordinates(
    recognition: Recognition,
    modelId: string,
    canvas: CanvasDimensions,
    scaling: ScalingParameters
  ): Coordinates {
    if (modelId.includes('detr') || modelId.includes('yolos')) {
      // Hugging Face models return [xmin, ymin, xmax, ymax] coordinates
      const [xmin, ymin, xmax, ymax] = recognition.bbox;
      
      // Convert from absolute coordinates to relative canvas coordinates
      const scaleX = canvas.width / scaling.scaledWidth;
      const scaleY = canvas.height / scaling.scaledHeight;
      
      return {
        x: xmin * scaleX,
        y: ymin * scaleY,
        width: (xmax - xmin) * scaleX,
        height: (ymax - ymin) * scaleY
      };
    } else {
      // COCO-SSD returns [x, y, width, height] coordinates
      const [x, y, width, height] = recognition.bbox;
      const scaleX = canvas.width / scaling.scaledWidth;
      const scaleY = canvas.height / scaling.scaledHeight;
      
      return {
        x: (x - scaling.offsetX) * scaleX,
        y: (y - scaling.offsetY) * scaleY,
        width: width * scaleX,
        height: height * scaleY
      };
    }
  }

  /**
   * Draws the bounding box for a recognition
   */
  function drawBoundingBox(
    ctx: CanvasRenderingContext2D,
    coords: Coordinates
  ) {
    const { x, y, width, height } = coords;
    
    ctx.fillStyle = 'rgba(75, 83, 32, 0.1)';
    ctx.fillRect(x, y, width, height);
    
    ctx.strokeStyle = '#4B5320';
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, width, height);
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
  }

  /**
   * Calculates label position
   */
  function calculateLabelPosition(
    coords: Coordinates,
    labelDimensions: { width: number; height: number },
    canvas: CanvasDimensions
  ): { x: number; y: number } {
    const padding = 8;
    const labelX = coords.x;
    const labelY = coords.y > labelDimensions.height + padding ? 
      coords.y - labelDimensions.height - padding : 
      coords.y + coords.height + padding;

    return {
      x: Math.max(0, Math.min(labelX, canvas.width - labelDimensions.width)),
      y: Math.max(labelDimensions.height, Math.min(labelY, canvas.height - padding))
    };
  }

  /**
   * Draws the label for a recognition
   */
  function drawLabel(
    ctx: CanvasRenderingContext2D,
    text: string,
    position: { x: number; y: number },
    width: number,
    height: number
  ) {
    const padding = 8;
    
    // Ensure position coordinates are valid numbers
    const x = Math.max(0, Number.isFinite(position.x) ? position.x : 0);
    const y = Math.max(0, Number.isFinite(position.y) ? position.y : 0);
    
    // Draw background
    ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
    ctx.shadowBlur = 8;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 2;

    try {
      const gradient = ctx.createLinearGradient(
        x, 
        y, 
        x + width, 
        y
      );
      gradient.addColorStop(0, '#4B5320');
      gradient.addColorStop(1, '#5B6330');
      ctx.fillStyle = gradient;
    } catch (error) {
      // Fallback to solid color if gradient creation fails
      console.error('Error creating gradient: ', error);
      ctx.fillStyle = '#4B5320';
    }

    ctx.beginPath();
    ctx.roundRect(x, y - height, width, height, 6);
    ctx.fill();

    // Draw text
    ctx.shadowColor = 'transparent';
    ctx.fillStyle = '#FFFFFF';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + padding, y - height/2);
  }

  /**
   * Draws a single recognition result on the canvas
   */
  function drawRecognitionResult(
    ctx: CanvasRenderingContext2D,
    recognition: Recognition,
    canvasDimensions: CanvasDimensions,
    modelId: string,
    scaling: ScalingParameters
  ) {
    const coords = convertCoordinates(recognition, modelId, canvasDimensions, scaling);
    const label = `${recognition.class} ${(recognition.score * 100).toFixed(1)}%`;
    
    // Set font for measuring and drawing
    ctx.font = 'bold 14px Inter, sans-serif';
    const padding = 8;
    const labelDimensions = {
      width: ctx.measureText(label).width + (padding * 2),
      height: 28
    };

    // Draw bounding box
    drawBoundingBox(ctx, coords);

    // Calculate label position
    const labelPosition = calculateLabelPosition(
      coords,
      labelDimensions,
      canvasDimensions
    );

    // Draw label background and text
    drawLabel(ctx, label, labelPosition, labelDimensions.width, labelDimensions.height);
  }

  /**
   * Handles downloading the processed image
   */
  const handleDownload = () => {
    if (!processedImage) return;

    const link = document.createElement('a');
    link.href = processedImage;
    link.download = 'detected-objects.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-4 min-h-[calc(100vh-10rem)]">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6 bg-white/80 backdrop-blur-sm p-4 sm:p-6 rounded-xl shadow-sm">
        <div className="flex-1 min-w-0">
          <Link
            href="/"
            className="inline-flex items-center text-[#4B5320] hover:text-[#5B6330] group text-sm font-medium"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1 transition-transform group-hover:-translate-x-1" />
            Back to Home
          </Link>
          <div className="flex items-center mt-2">
            <CubeTransparentIcon className="h-7 w-7 mr-2 text-[#4B5320]" />
            <h1 className="text-2xl font-bold bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-transparent bg-clip-text">
              Object Recognition
            </h1>
          </div>
          <p className="mt-2 text-sm text-gray-600">
            Detect and locate objects in images using advanced AI models
          </p>
        </div>
        <div className="w-full sm:w-auto sm:max-w-md">
          <ModelSelector
            models={models.recognition}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 lg:gap-8 p-2 sm:p-4 h-[calc(100vh-20rem)]">
        <div className="w-full h-[calc(100vh-24rem)] min-h-[400px] flex flex-col items-center justify-center">
          <ImageUpload 
            onImageSelect={handleImageSelect}
            processedImage={processedImage}
            isProcessed={!!processedImage || hasNoRecognitions}
            onDownload={handleDownload}
          />
        </div>

        <div className="w-full h-[calc(100vh-21rem)] min-h-[400px]">
          <div className="relative top-20">
            <div className="flex-1 min-h-0 overflow-y-auto">
              <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-sm mb-4">
                <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                  <div className="flex items-center">
                    <div className="h-8 w-1 bg-gradient-to-b from-[#4B5320] to-[#5B6330] rounded-full mr-3" />
                    <h2 className="text-xl font-semibold bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-transparent bg-clip-text">
                      {recognitions ? 'Results' : 'Analysis'}
                    </h2>
                  </div>
                  {image && !processedImage && !hasNoRecognitions && (
                    <motion.button
                      onClick={processImage}
                      disabled={isProcessing}
                      className="w-full sm:w-auto inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-[#4B5320] hover:bg-[#5B6330] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#4B5320] disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:shadow-md"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      {isProcessing ? (
                        <>
                          <ArrowPathIcon className="animate-spin -ml-1 mr-2 h-5 w-5" />
                          Processing...
                        </>
                      ) : (
                        'Detect Objects'
                      )}
                    </motion.button>
                  )}
                </div>
              </div>

              <AnimatePresence mode="wait">
                {!image && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-white/80 backdrop-blur-sm rounded-lg p-6 text-center shadow-sm"
                  >
                    <CubeTransparentIcon className="h-12 w-12 mx-auto text-[#4B5320]/40 mb-4" />
                    <p className="text-gray-600 text-base mb-2">
                      Waiting for an image
                    </p>
                    <p className="text-sm text-gray-500">
                      Use the upload area on the left to get started
                    </p>
                  </motion.div>
                )}

                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="rounded-lg bg-red-50 p-4 mb-4"
                  >
                    <div className="flex">
                      <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                      </div>
                      <div className="ml-3">
                        <p className="text-sm font-medium text-red-800">{error}</p>
                      </div>
                    </div>
                  </motion.div>
                )}

                {!recognitions && !error && image && !processedImage && !hasNoRecognitions && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-white/80 backdrop-blur-sm rounded-lg p-6 text-center shadow-sm"
                  >
                    <CubeTransparentIcon className="h-12 w-12 mx-auto text-[#4B5320]/40 mb-4" />
                    <p className="text-gray-600 text-base mb-2">
                      Image loaded successfully
                    </p>
                    <p className="text-sm text-gray-500">
                      Press the &quot;Detect Objects&quot; button above to analyze
                    </p>
                  </motion.div>
                )}

                {recognitions && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white/80 backdrop-blur-sm shadow-lg rounded-xl overflow-hidden border border-[#4B5320]/10 max-h-[calc(100vh-12rem)] overflow-y-auto"
                  >
                    <div className="p-4 bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-white sticky top-0 z-10">
                      <h3 className="text-lg font-semibold">Detected Objects</h3>
                      <p className="text-sm text-white/80">
                        {recognitions.length} {recognitions.length === 1 ? 'object' : 'objects'} found with high confidence
                      </p>
                    </div>
                    <ul className="divide-y divide-gray-200">
                      {recognitions.map((recognition) => (
                        <motion.li
                          key={`${recognition.class}-${recognition.score}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.1 }}
                          className="p-4 hover:bg-gray-50 transition-colors"
                        >
                          <div className="flex justify-between items-start mb-2">
                            <div className="flex-1">
                              <span className="text-base font-medium text-gray-900 block mb-1">
                                {recognition.class}
                              </span>
                              <span className="text-sm text-gray-500">
                                Confidence Score
                              </span>
                            </div>
                            <span className="text-lg font-semibold bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-transparent bg-clip-text ml-4">
                              {(recognition.score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
                            <motion.div
                              className="h-full rounded-full bg-gradient-to-r from-[#4B5320] to-[#5B6330]"
                              initial={{ width: 0 }}
                              animate={{ width: `${recognition.score * 100}%` }}
                              transition={{ duration: 0.5, delay: 0.1 }}
                            />
                          </div>
                        </motion.li>
                      ))}
                    </ul>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 