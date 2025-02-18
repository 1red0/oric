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
  bbox: [number, number, number, number];
  class: string;
  score: number;
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
      // Load and prepare the image
      const img = new Image();
      img.src = image;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      // Create input canvas for model processing
      const inputCanvas = document.createElement('canvas');
      const inputCtx = inputCanvas.getContext('2d', { willReadFrequently: true });
      if (!inputCtx) throw new Error('Could not get input canvas context');

      // Set standard dimensions for model input
      const standardWidth = selectedModel.id === 'coco-ssd' ? 640 : 800; // Adjust size based on model
      const standardHeight = selectedModel.id === 'coco-ssd' ? 480 : 600;
      inputCanvas.width = standardWidth;
      inputCanvas.height = standardHeight;

      // Draw image maintaining aspect ratio
      const scale = Math.min(standardWidth / img.width, standardHeight / img.height);
      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;
      const offsetX = (standardWidth - scaledWidth) / 2;
      const offsetY = (standardHeight - scaledHeight) / 2;

      // Apply image preprocessing
      inputCtx.filter = 'contrast(1.2) brightness(1.1) saturate(1.2)'; // Enhanced color processing
      inputCtx.imageSmoothingEnabled = true;
      inputCtx.imageSmoothingQuality = 'high';

      // Clear and draw centered image
      inputCtx.fillStyle = '#000000';
      inputCtx.fillRect(0, 0, standardWidth, standardHeight);
      inputCtx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);

      // Apply advanced image processing
      const imageData = inputCtx.getImageData(0, 0, standardWidth, standardHeight);
      const data = imageData.data;
      
      // Enhanced image processing
      for (let i = 0; i < data.length; i += 4) {
        // Adaptive contrast enhancement
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        
        // Calculate luminance
        const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        
        // Adaptive contrast factor based on luminance
        const contrastFactor = luminance < 128 ? 1.3 : 1.1;
        
        // Apply contrast enhancement
        data[i] = Math.min(255, Math.max(0, ((r - 128) * contrastFactor) + 128));     // R
        data[i + 1] = Math.min(255, Math.max(0, ((g - 128) * contrastFactor) + 128)); // G
        data[i + 2] = Math.min(255, Math.max(0, ((b - 128) * contrastFactor) + 128)); // B
        
        // Edge enhancement
        if (i > 0 && i < data.length - 4) {
          const prevLuminance = 0.299 * data[i - 4] + 0.587 * data[i - 3] + 0.114 * data[i - 2];
          if (Math.abs(luminance - prevLuminance) > 20) {
            // Enhance edges
            data[i] = Math.min(255, data[i] * 1.2);
            data[i + 1] = Math.min(255, data[i + 1] * 1.2);
            data[i + 2] = Math.min(255, data[i + 2] * 1.2);
          }
        }
      }

      inputCtx.putImageData(imageData, 0, 0);

      // Apply sharpening filter
      inputCtx.filter = 'sharpen(1) contrast(1.1) brightness(1.05)';
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = standardWidth;
      tempCanvas.height = standardHeight;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) throw new Error('Could not get temp canvas context');
      tempCtx.drawImage(inputCanvas, 0, 0);
      inputCtx.drawImage(tempCanvas, 0, 0);

      // Create output canvas for final result
      const outputCanvas = document.createElement('canvas');
      outputCanvas.width = img.naturalWidth;
      outputCanvas.height = img.naturalHeight;
      const outputCtx = outputCanvas.getContext('2d');
      if (!outputCtx) throw new Error('Could not get output canvas context');

      // Draw original image on output canvas
      outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);

      // Get recognitions from ML model using the standardized input
      const modelInput = selectedModel.id === 'coco-ssd' ? 
        // Convert canvas to image for COCO-SSD
        Object.assign(new Image(), { src: inputCanvas.toDataURL() }) : 
        img;

      // Wait for the input image to load if we created a new one
      if (selectedModel.id === 'coco-ssd') {
        await new Promise((resolve, reject) => {
          modelInput.onload = resolve;
          modelInput.onerror = reject;
        });
      }

      const results = await detectObjects(
        selectedModel.id,
        modelInput,
        process.env.NEXT_PUBLIC_HF_TOKEN
      );

      const typedResults = results.map(result => ({
        ...result,
        bbox: result.bbox as [number, number, number, number]
      }));

      if (typedResults.length === 0) {
        setHasNoRecognitions(true);
        setError('No objects detected in the image. Try another image or a different model.');
        return;
      }

      setRecognitions(typedResults);

      // Draw recognitions with improved visuals
      typedResults.forEach(recognition => {
        let [x, y, width, height] = recognition.bbox;

        if (selectedModel.id === 'facebook/detr-resnet-50') {
          // DETR returns coordinates as [xmin, ymin, xmax, ymax] in relative format
          const [xmin, ymin, xmax, ymax] = recognition.bbox;
          x = xmin * outputCanvas.width;
          y = ymin * outputCanvas.height;
          width = (xmax - xmin) * outputCanvas.width;
          height = (ymax - ymin) * outputCanvas.height;
        } else {
          // COCO-SSD returns coordinates relative to the standardized input
          // Convert from input canvas coordinates to output canvas coordinates
          const scaleX = outputCanvas.width / scaledWidth;
          const scaleY = outputCanvas.height / scaledHeight;
          x = (x - offsetX) * scaleX;
          y = (y - offsetY) * scaleY;
          width *= scaleX;
          height *= scaleY;
        }

        // Ensure coordinates are within canvas bounds
        x = Math.max(0, Math.min(x, outputCanvas.width - width));
        y = Math.max(0, Math.min(y, outputCanvas.height - height));
        width = Math.min(width, outputCanvas.width - x);
        height = Math.min(height, outputCanvas.height - y);

        // Draw semi-transparent highlight
        outputCtx.fillStyle = 'rgba(75, 83, 32, 0.1)';
        outputCtx.fillRect(x, y, width, height);

        // Draw border with double line effect
        outputCtx.strokeStyle = '#4B5320';
        outputCtx.lineWidth = 4;
        outputCtx.strokeRect(x, y, width, height);
        outputCtx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        outputCtx.lineWidth = 2;
        outputCtx.strokeRect(x, y, width, height);

        // Prepare label text and styling
        const label = `${recognition.class} ${(recognition.score * 100).toFixed(1)}%`;
        outputCtx.font = 'bold 14px Inter, sans-serif';
        const textMetrics = outputCtx.measureText(label);
        const padding = 8;
        const labelWidth = textMetrics.width + (padding * 2);
        const labelHeight = 28;
        
        // Calculate optimal label position
        let labelX = x;
        let labelY = y > labelHeight + padding ? y - labelHeight - padding : y + height + padding;
        
        // Ensure label stays within canvas bounds
        labelX = Math.max(0, Math.min(labelX, outputCanvas.width - labelWidth));
        labelY = Math.max(labelHeight, Math.min(labelY, outputCanvas.height - padding));

        // Draw label background with shadow
        outputCtx.shadowColor = 'rgba(0, 0, 0, 0.2)';
        outputCtx.shadowBlur = 8;
        outputCtx.shadowOffsetX = 0;
        outputCtx.shadowOffsetY = 2;

        const gradient = outputCtx.createLinearGradient(labelX, labelY, labelX + labelWidth, labelY);
        gradient.addColorStop(0, '#4B5320');
        gradient.addColorStop(1, '#5B6330');

        outputCtx.fillStyle = gradient;
        outputCtx.beginPath();
        outputCtx.roundRect(labelX, labelY - labelHeight, labelWidth, labelHeight, 6);
        outputCtx.fill();

        // Draw label text
        outputCtx.shadowColor = 'transparent';
        outputCtx.fillStyle = '#FFFFFF';
        outputCtx.textBaseline = 'middle';
        outputCtx.fillText(label, labelX + padding, labelY - labelHeight/2);
      });

      setProcessedImage(outputCanvas.toDataURL('image/png'));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  };

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
        <div className="w-full h-[calc(100vh-24rem)] min-h-[400px] flex flex-col">
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