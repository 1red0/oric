'use client';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { CameraIcon, CubeTransparentIcon } from '@heroicons/react/24/outline';

const features = [
  {
    name: 'Image Classification',
    description: 'Identify objects, scenes, and concepts in images using state-of-the-art deep learning models from TensorFlow.js and Hugging Face.',
    details: 'Powered by MobileNet and ResNet-50 models',
    href: '/classification',
    icon: CameraIcon,
    color: 'from-[#4B5320] to-[#5B6330]'
  },
  {
    name: 'Object recognition',
    description: 'Detect and locate multiple objects within images with precise bounding boxes and confidence scores.',
    details: 'Using COCO-SSD and DETR models',
    href: '/recognition',
    icon: CubeTransparentIcon,
    color: 'from-[#5B6330] to-[#6B7340]'
  }
];

export default function Home() {
  return (
    <div className="min-h-[calc(100vh-10rem)] bg-gradient-to-b from-gray-50 to-white">
      <div className="max-w-6xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <motion.h1 
            className="text-5xl font-bold text-gray-900 mb-6"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            ORIC
          </motion.h1>
          <motion.p 
            className="text-2xl font-medium text-gray-800 mb-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            Object Recognition & Image Classification
          </motion.p>
          <motion.p 
            className="text-lg text-gray-600 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            A modern web application demonstrating the power of computer vision using TensorFlow.js and Hugging Face models. 
            Process images directly in your browser with state-of-the-art deep learning models.
          </motion.p>
        </motion.div>

        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:gap-16">
          {features.map((feature, index) => (
            <motion.div
              key={feature.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.2 + 0.5 }}
            >
              <Link
                href={feature.href}
                className="block h-full"
              >
                <motion.div
                  whileHover={{ scale: 1.02, y: -5 }}
                  whileTap={{ scale: 0.98 }}
                  className="relative h-full overflow-hidden rounded-2xl shadow-lg transition-all duration-200 hover:shadow-xl"
                >
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-90`} />
                  <div className="relative h-full p-8">
                    <feature.icon
                      className="h-12 w-12 text-white mb-6"
                      aria-hidden="true"
                    />
                    <h2 className="text-2xl font-semibold text-white mb-4">
                      {feature.name}
                    </h2>
                    <p className="text-lg text-gray-100 mb-4">
                      {feature.description}
                    </p>
                    <p className="text-sm text-gray-200 mb-8">
                      {feature.details}
                    </p>
                    <div className="absolute bottom-8 right-8">
                      <span className="inline-flex items-center text-white font-medium">
                        Try it now
                        <svg
                          className="ml-2 h-5 w-5 transition-transform group-hover:translate-x-1"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M13 7l5 5m0 0l-5 5m5-5H6"
                          />
                        </svg>
                      </span>
                    </div>
                  </div>
                </motion.div>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}
