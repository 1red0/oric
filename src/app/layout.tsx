import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navigation from '@/components/Navigation';

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ORIC - Object Recognition & Image Classification",
  description: "A Next.js application for object recognition and image classification using TensorFlow.js and Hugging Face",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gradient-to-b from-gray-100 to-white min-h-screen`}>
        <Navigation />
        <main className="w-[90%] max-w-[1920px] mx-auto px-2 sm:px-4 lg:px-6 py-4 sm:py-6 lg:py-8 min-h-[calc(100vh-4rem)]">
          {children}
        </main>
      </body>
    </html>
  );
}
