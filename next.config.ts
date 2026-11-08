import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    resolveAlias: {
      sharp: false,
      "onnxruntime-node": false,
    },
  },
};

export default nextConfig;
