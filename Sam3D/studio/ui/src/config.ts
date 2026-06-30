// Runtime config: prefer the window.__SAM3D_CONFIG__ injected by deploy.sh
// (config.js on CloudFront), fall back to Vite env vars for local dev.
export interface AppConfig {
  region: string;
  userPoolId: string;
  userPoolClientId: string;
  apiUrl: string; // ends with "/"
}

const fromWindow = window.__SAM3D_CONFIG__;

export const config: AppConfig = fromWindow ?? {
  region: import.meta.env.VITE_REGION ?? "",
  userPoolId: import.meta.env.VITE_USER_POOL_ID ?? "",
  userPoolClientId: import.meta.env.VITE_USER_POOL_CLIENT_ID ?? "",
  apiUrl: import.meta.env.VITE_API_URL ?? "",
};

export function apiBase(): string {
  return config.apiUrl.endsWith("/") ? config.apiUrl : config.apiUrl + "/";
}
