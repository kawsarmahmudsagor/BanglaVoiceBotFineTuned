import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The floating dev badge covers the sidebar's Collapse button.
  devIndicators: false,
  // The Models page became Registry; keep old links working.
  async redirects() {
    return [
      { source: "/models", destination: "/registry", permanent: false },
      { source: "/models/:id", destination: "/registry/:id", permanent: false },
      // Recipes became Train; keep old links working.
      { source: "/recipes", destination: "/train", permanent: false },
      { source: "/recipes/:path*", destination: "/train/:path*", permanent: false },
    ];
  },
};

export default nextConfig;
