export const SLUG_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
export const isValidSlug = (s: string): boolean => SLUG_RE.test(s);
