import { Suspense } from "react";
import { PageHeader, Panel } from "@/components/matra";
import { UploadForm } from "@/components/upload-form";

/** Full-page upload, kept for deep links such as /datasets/new?task=asr from the catalog. */
export default function NewDatasetPage() {
  return (
    <div className="space-y-5">
      <PageHeader title="Upload dataset" />
      <Panel className="max-w-2xl">
        <Suspense fallback={null}>
          <UploadForm />
        </Suspense>
      </Panel>
    </div>
  );
}
