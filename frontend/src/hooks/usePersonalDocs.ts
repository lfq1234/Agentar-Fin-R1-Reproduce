import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteDocument,
  getDocumentStatus,
  listDocuments,
  uploadDocuments,
  ApiError,
} from "../api/client";
import type { PersonalDocument } from "../types/agent";

const ALLOWED_EXT = ["pdf", "docx", "txt", "md"];
const MAX_SIZE = 20 * 1024 * 1024; // 20MB
const MAX_FILES = 10;
const POLL_INTERVAL = 2000; // 2s
const POLL_MAX = 60; // 最多 60 次（约 2 分钟）

// 个人文档：上传 / 列表 / 解析状态轮询 / 删除（需求 FR1~FR3, FR9）。
export function usePersonalDocs(onParseDone?: () => void) {
  const [documents, setDocuments] = useState<PersonalDocument[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timers = useRef<Record<string, ReturnType<typeof setInterval>>>({});

  const stopPoll = useCallback((id: string) => {
    const t = timers.current[id];
    if (t !== undefined) {
      clearInterval(t);
      delete timers.current[id];
    }
  }, []);

  const fetchDocs = useCallback(async () => {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setError(null);
    } catch (e) {
      setError((e as ApiError).message ?? "获取文档列表失败");
    }
  }, []);

  // 解析状态轮询：2s 一次，done/error 或达上限后停止（评审 S2：含 cleanup）。
  const pollStatus = useCallback(
    (id: string) => {
      let count = 0;
      stopPoll(id);
      const t = setInterval(async () => {
        count += 1;
        try {
          const doc = await getDocumentStatus(id);
          setDocuments((prev) => prev.map((d) => (d.id === id ? doc : d)));
          if (doc.status === "done" || doc.status === "error" || count >= POLL_MAX) {
            stopPoll(id);
            if (doc.status === "done") onParseDone?.();
          }
        } catch {
          stopPoll(id);
        }
      }, POLL_INTERVAL);
      timers.current[id] = t;
    },
    [onParseDone, stopPoll],
  );

  const uploadFiles = useCallback(
    async (files: FileList) => {
      const arr = Array.from(files);
      if (arr.length === 0) return;
      const valid: File[] = [];
      const rejected: string[] = [];
      for (const f of arr) {
        const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
        if (!ALLOWED_EXT.includes(ext)) {
          rejected.push(`${f.name}：不支持的格式（仅 PDF/DOCX/TXT/MD）`);
          continue;
        }
        if (f.size > MAX_SIZE) {
          rejected.push(`${f.name}：超过 20MB`);
          continue;
        }
        valid.push(f);
      }
      if (valid.length > MAX_FILES) {
        rejected.push(`单次最多上传 ${MAX_FILES} 个文件`);
        valid.length = 0;
      }
      setError(rejected.length ? `已跳过：${rejected.join("；")}` : null);
      if (valid.length === 0) return;

      setUploading(true);
      try {
        const uploaded = await uploadDocuments(valid);
        setDocuments((prev) => [...uploaded, ...prev]);
        // 未完成的文档启动轮询
        for (const d of uploaded) {
          if (d.status === "pending" || d.status === "parsing") pollStatus(d.id);
        }
      } catch (e) {
        setError((e as ApiError).message ?? "上传失败");
      } finally {
        setUploading(false);
      }
    },
    [pollStatus],
  );

  const deleteDoc = useCallback(
    async (id: string) => {
      stopPoll(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      try {
        await deleteDocument(id);
      } catch (e) {
        setError((e as ApiError).message ?? "删除失败");
        fetchDocs();
      }
    },
    [fetchDocs, stopPoll],
  );

  useEffect(() => {
    fetchDocs();
    const snapshot = timers.current;
    return () => {
      Object.values(snapshot).forEach((t) => clearInterval(t));
    };
  }, [fetchDocs]);

  return { documents, uploading, error, setError, fetchDocs, uploadFiles, deleteDoc };
}
