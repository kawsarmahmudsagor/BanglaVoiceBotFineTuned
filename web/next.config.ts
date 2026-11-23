import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The Models page became Registry; keep old links working.
  async redirects() {
    return [
      { source: "/models", destination: "/registry", permanent: false },
      { source: "/models/:id", destination: "/registry/:id", permanent: false },
    ];
  },
};

export default nextConfig;
