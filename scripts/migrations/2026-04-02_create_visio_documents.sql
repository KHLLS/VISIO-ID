-- VISIO.ID — Migration: Create visio_documents table
-- Tanggal  : 2026-04-02
-- Author   : VISIO.ID Team
-- Deskripsi: Schema utama untuk menyimpan chunks + embeddings dari pipeline
--
-- CARA EKSEKUSI:
--   Buka Supabase SQL Editor → paste → Run
--   Atau via CLI: psql -h <host> -d postgres -f scripts/migrations/2026-04-02_create_visio_documents.sql

-- ─── Aktifkan extension pgvector ─────────────────────────────────────
create extension if not exists vector;

-- ─── Tabel utama ─────────────────────────────────────────────────────
create table if not exists visio_documents (
    id          bigserial primary key,
    chunk_id    text unique not null,       -- hash deterministik dari doc+index
    doc_id      text,                        -- hash dari URL sumber
    source      text,                        -- URL halaman asal
    category    text,                        -- product_page | blog | about | faq | general
    content     text,                        -- teks konten chunk
    language    text default 'id',           -- kode bahasa (ISO 639-1)
    metadata    jsonb,                       -- brand_name, industry, title, url, chunk_index, dll
    embedding   vector(768),                 -- multilingual-e5-base output
    created_at  timestamptz default now()
);

-- ─── Index untuk vector similarity search ────────────────────────────
-- Menggunakan IVFFlat dengan cosine distance
-- lists=100 cocok untuk ~100k–1M rows
create index if not exists visio_documents_embedding_idx
    on visio_documents
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- ─── Index tambahan untuk filter cepat ───────────────────────────────
create index if not exists visio_documents_metadata_brand_idx
    on visio_documents
    using gin (metadata jsonb_path_ops);

create index if not exists visio_documents_category_idx
    on visio_documents (category);

create index if not exists visio_documents_doc_id_idx
    on visio_documents (doc_id);

-- ─── Row Level Security (RLS) ─────────────────────────────────────────
-- Aktifkan RLS — hanya service_role key yang bisa write
alter table visio_documents enable row level security;

-- Policy: read-only untuk authenticated users
create policy "Public read visio_documents"
    on visio_documents
    for select
    to authenticated
    using (true);

-- Policy: full access untuk service role (backend pipeline)
create policy "Service role full access"
    on visio_documents
    for all
    to service_role
    using (true)
    with check (true);

-- ─── Verifikasi ───────────────────────────────────────────────────────
-- Setelah migrasi, jalankan ini untuk memverifikasi:
-- select count(*) from visio_documents;
-- select pg_size_pretty(pg_total_relation_size('visio_documents'));
