"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";

import { fetchCallDetail } from "@/lib/api";
import type { CallDetail } from "@/types";

export default function CallSessionPage() {
  const params = useParams<{ session_id: string }>();
  const router = useRouter();
  const sessionId = decodeURIComponent(params.session_id);

  const [detail, setDetail] = useState<CallDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setLoading(true);
    fetchCallDetail(sessionId)
      .then(setDetail)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sessionId]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3 text-sm text-slate-500">
        <button
          onClick={() => router.push("/calls")}
          className="hover:text-slate-700"
        >
          ← Back to calls
        </button>
      </div>

      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 p-8 animate-pulse">
          <div className="h-6 w-48 bg-slate-100 rounded mb-4" />
          <div className="h-4 w-64 bg-slate-100 rounded" />
        </div>
      )}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {detail && (
        <>
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <Detail label="Session" value={detail.session_id} mono />
            <Detail
              label="Phone"
              value={
                detail.farmer?.phone_number ??
                detail.record?.phone_number ??
                "—"
              }
              mono
            />
            <Detail label="Farmer" value={detail.farmer?.name ?? "—"} />
            <Detail
              label="When"
              value={
                detail.record?.timestamp
                  ? new Date(detail.record.timestamp).toLocaleString()
                  : "—"
              }
            />
            <Detail label="Location" value={detail.farmer?.location ?? "—"} />
            <Detail
              label="Language"
              value={detail.farmer?.language ?? "—"}
            />
            <Detail
              label="Duration"
              value={
                detail.record?.duration != null
                  ? `${detail.record.duration}s`
                  : "—"
              }
            />
            <Detail
              label="Recording"
              value={detail.record?.recording_path ? "Available" : "Not saved"}
            />
            {detail.farmer?.phone_number && (
              <div className="sm:col-span-2 lg:col-span-4">
                <Link
                  href={`/farmers/${encodeURIComponent(
                    detail.farmer.phone_number,
                  )}`}
                  className="text-sm font-medium text-blue-600 hover:text-blue-700"
                >
                  Open farmer profile →
                </Link>
              </div>
            )}
          </div>

          {detail.record?.recording_path && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
                Recording
              </div>
              <audio
                controls
                src={`/api/audio?path=${encodeURIComponent(
                  detail.record.recording_path,
                )}`}
                className="w-full h-10"
              />
            </div>
          )}

          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-8 py-6 border-b border-slate-100">
              <h3 className="text-base font-medium text-slate-800">
                Transcript
                <span className="ml-2 text-sm font-normal text-slate-400">
                  ({detail.transcript.length} messages)
                </span>
              </h3>
            </div>
            {detail.transcript.length === 0 ? (
              <div className="p-12 text-center text-sm text-slate-400">
                No transcript was captured for this session.
              </div>
            ) : (
              <ul className="divide-y divide-slate-100">
                {detail.transcript.map((m, i) => (
                  <li
                    key={i}
                    className="px-8 py-5 flex flex-col sm:flex-row gap-4"
                  >
                    <div className="sm:w-32 shrink-0">
                      <span
                        className={`text-xs font-semibold px-2.5 py-1 rounded-full border capitalize ${
                          m.role === "user"
                            ? "bg-blue-50 text-blue-700 border-blue-200"
                            : "bg-green-50 text-green-700 border-green-200"
                        }`}
                      >
                        {m.role === "user" ? "Caller" : "Assistant"}
                      </span>
                      <p className="text-xs text-slate-400 mt-1.5">
                        {m.timestamp
                          ? new Date(m.timestamp).toLocaleTimeString()
                          : ""}
                      </p>
                    </div>
                    <p className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed">
                      {m.message}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function Detail({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <span className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">
        {label}
      </span>
      <span className={`text-sm text-slate-900 ${mono ? "font-mono" : ""}`}>
        {value}
      </span>
    </div>
  );
}
