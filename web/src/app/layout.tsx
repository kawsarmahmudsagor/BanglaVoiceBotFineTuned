import type { Metadata } from "next";
import { Hind_Siliguri, JetBrains_Mono } from "next/font/google";
import { AppShell } from "@/components/shell/app-shell";
import { THEME_SCRIPT } from "@/lib/theme";
import "./globals.css";

// Drawn for both Bangla and Latin, so transcripts and labels share weight and height.
const uiSans = Hind_Siliguri({ variable: "--font-sans-ui", subsets: ["bengali", "latin"], weight: ["400", "500", "600"] });
// Only for IDs, metric values, hyperparameters, code and logs.
const uiMono = JetBrains_Mono({ variable: "--font-mono-ui", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: "Bangla Voice MLOps",
  description: "Datasets, recipes, training runs and models for Bangla voice.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`dark ${uiSans.variable} ${uiMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body
        className="text-sm antialiased"
      >
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
