import { Suspense } from "react";
import { UploadForm } from "@/components/upload-form";

export default function NewDatasetPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-lg font-semibold">Upload dataset</h1>
      <Suspense fallback={null}>
        <UploadForm />
      </Suspense>
    </div>
  );
}
