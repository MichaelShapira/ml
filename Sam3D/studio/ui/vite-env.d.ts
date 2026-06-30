/// <reference types="vite/client" />

// gaussian-splats-3d ships without TS types.
declare module "@mkkellogg/gaussian-splats-3d";

interface Window {
  __SAM3D_CONFIG__?: {
    region: string;
    userPoolId: string;
    userPoolClientId: string;
    apiUrl: string;
  };
}
