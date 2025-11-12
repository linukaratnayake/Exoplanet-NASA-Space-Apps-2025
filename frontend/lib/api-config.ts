// Utility helpers to resolve backend URLs across environments.

const normalizeUrl = (url: string) => {
  if (!url) {
    return url;
  }
  return /^https?:\/\//i.test(url) ? url : `https://${url}`;
};

export const ensureAbsoluteUrl = (
  value: string | undefined,
  fallback?: string
) => {
  const raw = value && value.trim().length > 0 ? value.trim() : fallback ?? "";
  if (!raw) {
    return "";
  }
  return normalizeUrl(raw);
};

export const getEnvVar = (key: string) => {
  const env = (globalThis as any)?.process?.env;
  return env?.[key] as string | undefined;
};

export const getBackendUrl = () => {
  if (typeof window === "undefined") {
    return ensureAbsoluteUrl(getEnvVar("BACKEND_URL"), "http://localhost:8000");
  }

  // Client-side: use Next.js API routes as proxy
  return "";
};

export const getApiUrl = (endpoint: string) => {
  if (typeof window === "undefined") {
    const backendUrl = getBackendUrl();
    return `${backendUrl}${endpoint}`;
  }

  return `/api/proxy${endpoint}`;
};
