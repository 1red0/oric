'use client';

import { useCallback, useState, useRef } from 'react';
import { useDropzone, FileRejection } from 'react-dropzone';
import { PhotoIcon, XMarkIcon, ArrowUpTrayIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';

interface ImageUploadProps {
  readonly onImageSelect: (imageData: string | null) => void;
  readonly maxSizeMB?: number;
  readonly processedImage?: string | null;
  readonly isProcessed?: boolean;
  readonly onDownload?: () => void;
}

const DEFAULT_MAX_SIZE = 5; // 5MB

const dropzoneAnimation = {
  initial: { opacity: 0, scale: 0.95 },
  animate: { opacity: 1, scale: 1 },
  exit: { opacity: 0, scale: 0.95 },
  transition: { duration: 0.2 }
};

export default function ImageUpload({ 
  onImageSelect, 
  maxSizeMB = DEFAULT_MAX_SIZE,
  processedImage = null,
  isProcessed = false,
  onDownload
}: ImageUploadProps) {
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const previewRef = useRef<HTMLImageElement>(null);

  const clearImage = () => {
    setPreview(null);
    onImageSelect(null);
  };

  const handleImageLoad = () => {
    if (previewRef.current) {
      const { naturalWidth, naturalHeight } = previewRef.current;
      if (naturalWidth < 224 || naturalHeight < 224) {
        setError('Image dimensions must be at least 224x224 pixels');
        clearImage();
        return;
      }
    }
  };

  const processFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        const imageData = e.target.result as string;
        setPreview(imageData);
        onImageSelect(imageData);
      }
    };
    reader.onerror = () => {
      setError('Error reading file');
    };
    reader.readAsDataURL(file);
  };

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: FileRejection[]) => {
    setError(null);
    setIsDragging(false);

    if (fileRejections.length > 0) {
      const errors = fileRejections[0].errors;
      if (errors.some((e) => e.code === 'file-too-large')) {
        setError(`File is too large. Maximum size is ${maxSizeMB}MB`);
      } else if (errors.some((e) => e.code === 'file-invalid-type')) {
        setError('Invalid file type. Please upload an image file');
      } else {
        setError('Error uploading file');
      }
      return;
    }

    const file = acceptedFiles[0];
    if (file) {
      processFile(file);
    }
  }, [maxSizeMB, processFile]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    },
    maxSize: maxSizeMB * 1024 * 1024,
    multiple: false,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false)
  });

  const displayImage = processedImage ?? preview;

  return (
    <div className="space-y-4">
      <AnimatePresence mode="wait">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="rounded-md bg-red-50 p-4 shadow-sm"
          >
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3 flex-1">
                <p className="text-sm font-medium text-red-800">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="ml-auto flex-shrink-0 text-red-700 hover:text-red-900 transition-colors"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <AnimatePresence mode="wait">
        {!displayImage ? (
          <motion.div
            {...dropzoneAnimation}
            className="relative h-full"
          >
            <div
              {...getRootProps()}
              className={`relative h-full flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl transition-all duration-200 bg-white/80 backdrop-blur-sm z-10 ${
                isDragging
                  ? 'border-[#4B5320] bg-[#4B5320]/5'
                  : 'border-gray-300 hover:border-[#4B5320]/50'
              } ${isProcessed ? 'border-none p-0' : ''}`}
            >
              <input {...getInputProps()} />
              <motion.div
                animate={isDragging ? { scale: 1.1, y: -10 } : { scale: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
              >
                <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-base text-gray-600 font-medium">
                  {isDragging ? 'Drop the image here' : 'Drag & drop an image here, or click to select'}
                </p>
                <p className="mt-1 text-sm text-gray-500">
                  Supported formats: PNG, JPEG, GIF, WebP (max {maxSizeMB}MB)
                </p>
              </motion.div>
              {isDragging && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="absolute inset-0 pointer-events-none"
                >
                  <div className="absolute inset-4 border-2 border-[#4B5320] rounded-lg border-dashed animate-pulse" />
                </motion.div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div
            {...dropzoneAnimation}
            className="relative h-[calc(100vh-20rem)] rounded-xl overflow-hidden bg-gray-50 shadow-lg group"
          >
            <div className="relative w-full h-full flex items-center justify-center">
              <img
                ref={previewRef}
                src={displayImage}
                alt="Preview"
                onLoad={handleImageLoad}
                className="max-w-full max-h-full w-auto h-auto object-contain"
              />
            </div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300"
            >
              <div className="absolute inset-0 flex items-center justify-center space-x-4 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                {isProcessed ? (
                  <>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={clearImage}
                      className="px-4 py-2 bg-white text-gray-900 rounded-lg font-medium hover:bg-gray-100 transition-colors shadow-lg hover:shadow-xl flex items-center space-x-2"
                    >
                      <ArrowPathIcon className="h-5 w-5" />
                      <span>Try New Image</span>
                    </motion.button>
                    {onDownload && (
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={onDownload}
                        className="px-4 py-2 bg-[#4B5320] text-white rounded-lg font-medium hover:bg-[#5B6330] transition-colors shadow-lg hover:shadow-xl flex items-center space-x-2"
                      >
                        <ArrowUpTrayIcon className="h-5 w-5" />
                        <span>Save Result</span>
                      </motion.button>
                    )}
                  </>
                ) : (
                  <>
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={() => {
                        const input = document.createElement('input');
                        input.type = 'file';
                        input.accept = 'image/*';
                        input.onchange = (e) => {
                          const file = (e.target as HTMLInputElement).files?.[0];
                          if (file) processFile(file);
                        };
                        input.click();
                      }}
                      className="px-4 py-2 bg-white text-gray-900 rounded-lg font-medium hover:bg-gray-100 transition-colors shadow-lg hover:shadow-xl flex items-center space-x-2"
                    >
                      <ArrowUpTrayIcon className="h-5 w-5" />
                      <span>Change Image</span>
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.05, rotate: 90 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={clearImage}
                      className="p-2 bg-white text-gray-900 rounded-lg hover:bg-gray-100 transition-colors shadow-lg hover:shadow-xl"
                    >
                      <XMarkIcon className="h-5 w-5" />
                    </motion.button>
                  </>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
} 