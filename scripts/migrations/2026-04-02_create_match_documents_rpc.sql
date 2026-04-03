-- VISIO.ID — Migration: match_documents RPC function
-- Tanggal  : 2026-04-02
-- Deskripsi: PostgreSQL function untuk vector similarity search via Supabase RPC
--
-- CARA EKSEKUSI:
--   Buka Supabase SQL Editor → paste → Run
--   Wajib dijalankan SETELAH 2026-04-02_create_visio_documents.sql

-- ─── Function: match_documents ────────────────────────────────────────
-- Dipanggil oleh Stage 4 RAG via: client.rpc("match_documents", {...})
create or replace function match_documents(
    query_embedding  vector(768),
    match_threshold  float   default 0.3,
    match_count      int     default 20
)
returns table (
    id          bigint,
    chunk_id    text,
    doc_id      text,
    source      text,
    category    text,
    content     text,
    language    text,
    metadata    jsonb,
    similarity  float
)
language sql stable
as $$
    select
        id,
        chunk_id,
        doc_id,
        source,
        category,
        content,
        language,
        metadata,
        1 - (embedding <=> query_embedding) as similarity
    from visio_documents
    where 1 - (embedding <=> query_embedding) > match_threshold
    order by similarity desc
    limit match_count;
$$;

-- Grant akses ke role yang dipakai
grant execute on function match_documents to authenticated;
grant execute on function match_documents to service_role;

-- ─── Verifikasi ───────────────────────────────────────────────────────
-- Test function setelah dijalankan:
-- select * from match_documents(
--     array_fill(0::float, array[768])::vector,
--     0.0,
--     5
-- );
