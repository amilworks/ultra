import { useEffect, useMemo, useState } from "react";
import { fileExt, formatBytes } from "../lib/format";
import type { UploadedFileRecord } from "../types";
import { FileUpload, FileUploadContent, FileUploadTrigger } from "./prompt-kit";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type FilePanelProps = {
  pendingFiles: File[];
  uploadedFiles: UploadedFileRecord[];
  uploading: boolean;
  onAddFiles: (files: File[]) => void;
  onRemovePending: (index: number) => void;
  onClearPending: () => void;
  onUpload: () => Promise<void>;
  onDropUploaded: (fileId: string) => void;
};

type Preview = {
  url: string;
  name: string;
};

const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "bmp", "webp"]);

export function FilePanel(props: FilePanelProps) {
  const [pickerKey, setPickerKey] = useState(0);
  const previews = useMemo<Preview[]>(() => {
    return props.pendingFiles
      .filter((file) => IMAGE_EXTENSIONS.has(fileExt(file.name)))
      .slice(0, 3)
      .map((file) => ({ name: file.name, url: URL.createObjectURL(file) }));
  }, [props.pendingFiles]);

  useEffect(() => {
    return () => {
      previews.forEach((preview) => URL.revokeObjectURL(preview.url));
    };
  }, [previews]);

  return (
    <Card className="border-white/60 bg-white/82 shadow-xl backdrop-blur-sm">
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-3">
        <CardTitle>Uploads</CardTitle>
        <Badge variant="secondary">{props.uploadedFiles.length} ready</Badge>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        <p className="text-muted-foreground text-sm">
          Add local files. They are copied to server session storage before chat.
        </p>
        <FileUpload
          key={pickerKey}
          onFilesAdded={(files) => {
            props.onAddFiles(files);
            setPickerKey((value) => value + 1);
          }}
          multiple
        >
          <FileUploadContent>
            <p className="text-muted-foreground text-sm">
              Drop files here or use the picker to add data for this session.
            </p>
          </FileUploadContent>
          <FileUploadTrigger asChild>
            <Button variant="outline" size="sm">
              Choose files
            </Button>
          </FileUploadTrigger>
        </FileUpload>
        {props.pendingFiles.length > 0 ? (
          <div className="flex flex-col gap-2">
            <div className="list-header">
              <strong>Selected files</strong>
              <Button variant="ghost" size="sm" onClick={props.onClearPending}>
                Clear
              </Button>
            </div>
            <ul className="list">
              {props.pendingFiles.map((file, index) => (
                <li key={`${file.name}-${file.size}-${index}`} className="list-row">
                  <span className="file-name">{file.name}</span>
                  <span className="file-meta">{formatBytes(file.size)}</span>
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={() => props.onRemovePending(index)}
                    aria-label={`Remove ${file.name}`}
                  >
                    Remove
                  </Button>
                </li>
              ))}
            </ul>
            {previews.length > 0 ? (
              <div className="preview-grid">
                {previews.map((preview) => (
                  <figure key={preview.url} className="preview-card">
                    <img src={preview.url} alt={preview.name} />
                    <figcaption>{preview.name}</figcaption>
                  </figure>
                ))}
              </div>
            ) : null}
            <Button disabled={props.uploading} onClick={() => void props.onUpload()}>
              {props.uploading ? "Uploading..." : "Upload selected files"}
            </Button>
          </div>
        ) : null}
        {props.uploadedFiles.length > 0 ? (
          <div className="flex flex-col gap-2">
            <strong>Available to tools</strong>
            <ul className="list">
              {props.uploadedFiles.map((file) => (
                <li key={file.file_id} className="list-row">
                  <span className="file-name">{file.original_name}</span>
                  <span className="file-meta">{formatBytes(file.size_bytes)}</span>
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={() => props.onDropUploaded(file.file_id)}
                    aria-label={`Remove ${file.original_name} from context`}
                  >
                    Exclude
                  </Button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
