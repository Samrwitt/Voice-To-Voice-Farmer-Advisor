"use client";

import { useEffect, useMemo, useState } from "react";
import {
  approveKBDocument,
  deleteKBDocument,
  fetchKBDocuments,
  reindexKBDocument,
  rejectKBDocument,
  uploadKBDocument,
} from "@/lib/api";
import { hasRole } from "@/lib/auth";
import type { KBDocument } from "@/types";
import Badge from "@/components/ui/Badge";

function statusVariant(status: string) {
  if (status === "approved") return "success";
  if (status === "uploaded") return "warning";
  if (status === "rejected") return "danger";
  return "neutral";
}

function indexVariant(status: string) {
  if (status === "indexed") return "success";
  if (status === "indexing") return "info";
  if (status === "failed") return "danger";
  if (status === "pending") return "warning";
  return "neutral";
}

export default function KBDocumentsPage() {
  const canWrite = typeof window !== "undefined" ? hasRole("admin", "expert") : false;
  const isAdmin = typeof window !== "undefined" ? hasRole("admin") : false;

  const [docs, setDocs] = useState<KBDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  // Upload form
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [topic, setTopic] = useState("");
  const [crop, setCrop] = useState("");
  const [region, setRegion] = useState("");
  const [category, setCategory] = useState("");
  const [uploading, setUploading] = useState(false);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(""), 3000);
  };

  const load = () => {
    setLoading(true);
    setError("");
    fetchKBDocuments()
      .then(setDocs)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const filtered = useMemo(() => {
    return docs;
  }, [docs]);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    try {
      await uploadKBDocument(file, {
        title: title.trim() || undefined,
        topic: topic.trim() || undefined,
        crop: crop.trim() || undefined,
        region: region.trim() || undefined,
        category: category.trim() || undefined,
      });
      setFile(null);
      setTitle("");
      setTopic("");
      setCrop("");
      setRegion("");
      setCategory("");
      showToast("Document uploaded ✓");
      load();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const act = async (id: string, fn: () => Promise<any>, msg: string) => {
    setBusyId(id);
    try {
      await fn();
      showToast(msg);
      load();
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Action failed");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="space-y-8">
      {toast && (
        <div className="fixed top-4 right-4 z-50 bg-slate-900 text-white text-sm font-medium px-4 py-2.5 rounded-lg shadow-lg">
          {toast}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mt-8">
        {/* Upload panel */}
        <div className="lg:col-span-4">
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden sticky top-8">
            <div className="px-8 py-6 border-b border-slate-100">
              <h3 className="text-base font-medium text-slate-800">
                Upload knowledge document
              </h3>
              <p className="text-sm text-slate-500 mt-1">
                TXT/MD/PDF supported.
              </p>
            </div>
            {canWrite ? (
              <form onSubmit={handleUpload} className="p-8 space-y-4">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-700">File</label>
                  <input
                    type="file"
                    accept=".txt,.md,.pdf,text/plain,text/markdown,application/pdf"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    className="block w-full text-sm text-slate-600"
                    required
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-slate-700">Title</label>
                  <input
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full h-10 px-3 rounded-md border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                    placeholder="Optional"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Topic</label>
                    <input
                      value={topic}
                      onChange={(e) => setTopic(e.target.value)}
                      className="w-full h-10 px-3 rounded-md border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                      placeholder="e.g. fertilizer"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Crop</label>
                    <input
                      value={crop}
                      onChange={(e) => setCrop(e.target.value)}
                      className="w-full h-10 px-3 rounded-md border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                      placeholder="e.g. Wheat"
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Region</label>
                    <input
                      value={region}
                      onChange={(e) => setRegion(e.target.value)}
                      className="w-full h-10 px-3 rounded-md border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                      placeholder="e.g. Oromia"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-sm font-medium text-slate-700">Category</label>
                    <input
                      value={category}
                      onChange={(e) => setCategory(e.target.value)}
                      className="w-full h-10 px-3 rounded-md border border-slate-300 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-green-500"
                      placeholder="e.g. agronomy"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={uploading || !file}
                  className="w-full h-10 bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-sm font-medium rounded-md transition-colors"
                >
                  {uploading ? "Uploading…" : "Upload"}
                </button>
              </form>
            ) : (
              <div className="p-8 text-center py-12">
                <span className="text-3xl block mb-3">🔒</span>
                <p className="text-sm font-medium text-slate-700">Read-only</p>
                <p className="text-sm text-slate-500 mt-1">
                  Only admins or experts can upload documents.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Docs list */}
        <div className="lg:col-span-8">
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-8 py-6 border-b border-slate-100 flex items-center justify-between">
              <h3 className="text-base font-medium text-slate-800">
                Knowledge documents
              </h3>
              <Badge label={`${docs.length} Total`} variant="neutral" />
            </div>

            {loading && (
              <div className="p-8 space-y-3 animate-pulse">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="h-16 bg-slate-100 rounded-md" />
                ))}
              </div>
            )}
            {error && (
              <div className="m-8 bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {!loading && !error && (
              <div className="divide-y divide-slate-100">
                {filtered.length === 0 ? (
                  <div className="py-16 text-center text-sm text-slate-400">
                    No documents uploaded yet.
                  </div>
                ) : (
                  filtered.map((d) => (
                    <div key={d.id} className="p-8 hover:bg-slate-50 transition-colors">
                      <div className="flex items-start justify-between gap-4">
                        <div className="space-y-2 min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="text-sm font-medium text-slate-900 truncate">
                              {d.title ?? d.filename}
                            </span>
                            <Badge label={d.status} variant={statusVariant(d.status)} />
                            <Badge label={d.indexing_status} variant={indexVariant(d.indexing_status)} />
                            <Badge label={`${d.chroma_doc_count} chunks`} variant="neutral" />
                          </div>
                          <p className="text-xs text-slate-500 font-mono break-all">
                            {d.filename} · {d.id}
                          </p>
                          <p className="text-xs text-slate-500">
                            {(d.topic || "—")} · {(d.crop || "—")} · {(d.region || "—")} · {(d.category || "—")}
                          </p>
                          {d.indexing_error && (
                            <div className="mt-2 bg-red-50 border border-red-200 rounded-md p-3 text-xs text-red-700">
                              {d.indexing_error}
                            </div>
                          )}
                        </div>

                        <div className="flex flex-col gap-2 shrink-0">
                          {canWrite && d.status !== "approved" && (
                            <button
                              onClick={() => act(d.id, () => approveKBDocument(d.id), "Approved + indexed ✓")}
                              disabled={busyId === d.id}
                              className="h-9 px-3 rounded-md bg-green-600 hover:bg-green-700 disabled:opacity-50 text-white text-xs font-medium"
                            >
                              Approve & index
                            </button>
                          )}
                          {canWrite && (
                            <button
                              onClick={() => act(d.id, () => reindexKBDocument(d.id), "Reindexed ✓")}
                              disabled={busyId === d.id}
                              className="h-9 px-3 rounded-md bg-slate-900 hover:bg-slate-800 disabled:opacity-50 text-white text-xs font-medium"
                            >
                              Re-index
                            </button>
                          )}
                          {canWrite && d.status !== "rejected" && (
                            <button
                              onClick={() => act(d.id, () => rejectKBDocument(d.id), "Rejected ✓")}
                              disabled={busyId === d.id}
                              className="h-9 px-3 rounded-md bg-amber-500 hover:bg-amber-600 disabled:opacity-50 text-white text-xs font-medium"
                            >
                              Reject
                            </button>
                          )}
                          {isAdmin && (
                            <button
                              onClick={() => act(d.id, () => deleteKBDocument(d.id), "Deleted ✓")}
                              disabled={busyId === d.id}
                              className="h-9 px-3 rounded-md bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white text-xs font-medium"
                            >
                              Delete
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

