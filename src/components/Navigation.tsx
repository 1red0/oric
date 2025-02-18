'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';

export default function Navigation() {
  const pathname = usePathname();
  
  const getAlternateLink = () => {
    if (pathname === '/classification') {
      return {
        href: '/recognition',
        text: 'Try Object Recognition'
      };
    } else if (pathname === '/recognition') {
      return {
        href: '/classification',
        text: 'Try Image Classification'
      };
    }
    return null;
  };

  const alternateLink = getAlternateLink();

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="bg-gradient-to-r from-[#4B5320] to-[#5B6330] text-white shadow-lg sticky top-0 z-50"
    >
      <div className="w-[95%] max-w-[1920px] mx-auto h-16 px-2 sm:px-4 lg:px-6 flex items-center justify-between">
        <Link href="/" className="flex items-center space-x-2 group">
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center"
          >
            <motion.span
              className="font-bold text-2xl bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-100"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              ORIC
            </motion.span>
            <motion.span
              className="ml-2 text-sm font-medium text-gray-200 opacity-0 group-hover:opacity-100 transition-all duration-300"
              initial={{ x: -10 }}
              animate={{ x: 0 }}
            >
              Home
            </motion.span>
          </motion.div>
        </Link>
        
        <AnimatePresence mode="wait">
          {alternateLink && pathname !== '/' && (
            <motion.div
              initial={{ opacity: 0, x: 20, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: -20, scale: 0.9 }}
              transition={{ type: "spring", stiffness: 200, damping: 20 }}
            >
              <Link
                href={alternateLink.href}
                className="group inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium 
                         bg-white/10 hover:bg-white/20 transition-all duration-300
                         border border-white/20 hover:border-white/30 hover:shadow-lg"
              >
                <motion.span
                  initial={{ x: 10 }}
                  animate={{ x: 0 }}
                  transition={{ delay: 0.1 }}
                >
                  {alternateLink.text}
                </motion.span>
                <motion.svg
                  className="ml-2 h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  whileHover={{ x: 5 }}
                  transition={{ type: "spring", stiffness: 400, damping: 10 }}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M13 7l5 5m0 0l-5 5m5-5H6"
                  />
                </motion.svg>
              </Link>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.nav>
  );
} 