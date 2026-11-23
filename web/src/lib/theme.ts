export type Theme = "light" | "dark";

const KEY = "theme";

function defaultStorage(): Storage | undefined {
  try {
    return typeof window === "undefined" ? undefined : window.localStorage;
  } catch {
    return undefined;
  }
}

/** The stored theme; dark when nothing valid is stored or storage is blocked. */
export function readTheme(storage: Pick<Storage, "getItem"> | undefined = defaultStorage()): Theme {
  try {
    return storage?.getItem(KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function writeTheme(t: Theme, storage: Pick<Storage, "setItem"> | undefined = defaultStorage()): void {
  try {
    storage?.setItem(KEY, t);
  } catch {
    // Blocked storage: the choice lasts for this page only.
  }
}

export function applyTheme(t: Theme): void {
  const root = document.documentElement;
  root.classList.toggle("dark", t === "dark");
  root.style.colorScheme = t;
}

/** Runs before first paint so the page never flashes the wrong theme. */
export const THEME_SCRIPT = `(function(){var d=true;try{d=localStorage.getItem('theme')!=='light'}catch(e){}var r=document.documentElement;r.classList.toggle('dark',d);r.style.colorScheme=d?'dark':'light'})()`;
