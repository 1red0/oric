'use client';

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ImageUpload from '@/components/ImageUpload';
import ModelSelector from '@/components/ModelSelector';
import { models, classifyImage } from '@/utils/ml-models';
import { ArrowPathIcon, ArrowLeftIcon, CameraIcon } from '@heroicons/react/24/outline';
import Link from 'next/link';

/**
 * Interface for classification results
 */
interface Classification {
  className: string;
  probability: number;
}

/**
 * ImageClassification component handles the classification of images using
 * various ML models (MobileNet from TensorFlow.js or ResNet from Hugging Face).
 */
export default function ImageClassification() {
  // State management
  const [selectedModel, setSelectedModel] = useState(models.classification[0]);
  const [image, setImage] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Classification[] | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasNoClassifications, setHasNoClassifications] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);

  /**
   * Handles image selection and resets related states
   */
  const handleImageSelect = (imageData: string | null) => {
    setImage(imageData);
    setPredictions(null);
    setError(null);
    setHasNoClassifications(false);
  };

  /**
   * Processes the selected image for classification
   */
  const processImage = async () => {
    if (!image) return;

    setIsProcessing(true);
    setError(null);
    setPredictions(null);
    setHasNoClassifications(false);

    try {
      const results = await classifyImage(
        selectedModel.id,
        image
      );

      if (!results || results.length === 0) {
        setHasNoClassifications(true);
        setError('No classifications found. Try another image or a different model.');
        return;
      }

      setPredictions(results.map(result => ({
        className: result.label,
        probability: result.score
      })));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
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
            <CameraIcon className="h-7 w-7 mr-2 text-[#4B5320]" />
            <h1 className="text-2xl font-bold bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-transparent bg-clip-text">
              Image Classification
            </h1>
          </div>
          <p className="mt-2 text-sm text-gray-600">
            Identify objects and scenes in images using advanced AI models
          </p>
        </div>
        <div className="w-full sm:w-auto sm:max-w-md">
          <ModelSelector
            models={models.classification}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            disabled={!!image}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6 lg:gap-8 p-2 sm:p-4 h-[calc(100vh-20rem)]">
        <div className="w-full h-[calc(100vh-24rem)] min-h-[400px] flex flex-col items-center justify-center">
          <ImageUpload 
            onImageSelect={handleImageSelect}
            isProcessed={!!predictions || hasNoClassifications}
          />
          {image && (
            <div className="mt-4 hidden">
              <img
                ref={imageRef}
                src={image}
                alt="Selected"
                className="w-full h-full object-contain"
              />
            </div>
          )}
        </div>

        <div className="w-full h-[calc(100vh-21rem)] min-h-[400px]">
          <div className="relative top-20">
            <div className="flex-1 min-h-0 overflow-y-auto">
              <div className="bg-white/80 backdrop-blur-sm p-4 rounded-xl shadow-sm mb-4">
                <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                  <div className="flex items-center">
                    <div className="h-8 w-1 bg-gradient-to-b from-[#4B5320] to-[#5B6330] rounded-full mr-3" />
                    <h2 className="text-xl font-semibold bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-transparent bg-clip-text">
                      {predictions ? 'Results' : 'Analysis'}
                    </h2>
                  </div>
                  {image && !predictions && !hasNoClassifications && (
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
                        'Classify Image'
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
                    <CameraIcon className="h-12 w-12 mx-auto text-[#4B5320]/40 mb-4" />
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

                {predictions && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white/80 backdrop-blur-sm shadow-lg rounded-xl overflow-hidden border border-[#4B5320]/10 max-h-[calc(100vh-12rem)] overflow-y-auto"
                  >
                    <div className="p-4 bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-white sticky top-0 z-10">
                      <h3 className="text-lg font-semibold">Classification Results</h3>
                      <p className="text-sm text-white/80">
                        {predictions.length} {predictions.length === 1 ? 'match' : 'matches'} found with high confidence
                      </p>
                    </div>
                    <ul className="divide-y divide-gray-200">
                      {predictions.map((prediction) => (
                        <motion.li
                          key={`${prediction.className}-${prediction.probability}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.1 }}
                          className="p-4 hover:bg-gray-50 transition-colors"
                        >
                          <div className="flex justify-between items-start mb-2">
                            <div className="flex-1">
                              <span className="text-base font-medium text-gray-900 block mb-1">
                                {prediction.className}
                              </span>
                              <span className="text-sm text-gray-500">
                                Confidence Score
                              </span>
                            </div>
                            <span className="text-lg font-semibold bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-transparent bg-clip-text ml-4">
                              {(prediction.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
                            <motion.div
                              className="h-full rounded-full bg-gradient-to-r from-[#4B5320] to-[#5B6330]"
                              initial={{ width: 0 }}
                              animate={{ width: `${prediction.probability * 100}%` }}
                              transition={{ duration: 0.5, delay: 0.1 }}
                            />
                          </div>
                        </motion.li>
                      ))}
                    </ul>
                  </motion.div>
                )}

                {!predictions && !error && image && !hasNoClassifications && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="bg-white/80 backdrop-blur-sm rounded-lg p-6 text-center shadow-sm"
                  >
                    <CameraIcon className="h-12 w-12 mx-auto text-[#4B5320]/40 mb-4" />
                    <p className="text-gray-600 text-base mb-2">
                      Image loaded successfully
                    </p>
                    <p className="text-sm text-gray-500">
                      Press the &quot;Classify Image&quot; button above to analyze
                    </p>
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