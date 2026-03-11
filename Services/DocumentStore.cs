using Microsoft.Data.Sqlite;

namespace PdfPal.Services;

// ── Public DTOs ────────────────────────────────────────────────────────────

public sealed record DocumentInfo(long Id, string Filename, string UploadedAt, int ChunkCount);

public sealed record CandidateChunk(
    long    DocumentId,
    string  DocumentName,
    int     PageNumber,
    int     ChunkIndex,
    string  Text,
    float[] Embedding,
    float   YStart,
    float   YEnd);

public sealed record RetrievedChunk(
    long   DocumentId,
    string DocumentName,
    int    PageNumber,
    int    ChunkIndex,
    string Text,
    float  YStart,
    float  YEnd);

// ── DocumentStore ──────────────────────────────────────────────────────────

public class DocumentStore
{
    private readonly string _dbPath;
    private readonly string _pdfDir;
    private readonly ILogger<DocumentStore> _logger;

    public DocumentStore(IConfiguration config, ILogger<DocumentStore> logger)
    {
        _dbPath = config["Database:Path"] ?? "/app/data/pdfpal.db";
        _pdfDir = Path.Combine(Path.GetDirectoryName(_dbPath)!, "pdfs");
        _logger = logger;
        Initialize();
    }

    // ── Init / reset ───────────────────────────────────────────────────────

    private void Initialize()
    {
        var dir = Path.GetDirectoryName(_dbPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        Wipe();
        CreateSchema();
        _logger.LogInformation("DocumentStore initialized at {Path}", _dbPath);
    }

    public void Reset()
    {
        Wipe();
        CreateSchema();
        _logger.LogInformation("Database reset — new session started");
    }

    public void Wipe()
    {
        // Flush all pooled connections before deleting — prevents stale file-descriptor
        // handles being handed back by the connection pool after the file is replaced.
        SqliteConnection.ClearAllPools();

        // DB files
        foreach (var suffix in new[] { "", "-wal", "-shm" })
        {
            var path = _dbPath + suffix;
            if (File.Exists(path))
                try { File.Delete(path); } catch (Exception ex) { _logger.LogWarning(ex, "Could not delete {Path}", path); }
        }

        // Stored PDF files
        if (Directory.Exists(_pdfDir))
            try { Directory.Delete(_pdfDir, recursive: true); } catch (Exception ex) { _logger.LogWarning(ex, "Could not delete pdf dir"); }
    }

    private void CreateSchema()
    {
        using var conn = Open();
        using var cmd  = conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY,
                filename    TEXT    NOT NULL,
                uploaded_at TEXT    NOT NULL,
                chunk_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id),
                page_number INTEGER,
                chunk_index INTEGER,
                text        TEXT,
                embedding   BLOB,
                y_start     REAL    DEFAULT 0,
                y_end       REAL    DEFAULT 1
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content = 'chunks',
                content_rowid = 'id'
            );
            """;
        cmd.ExecuteNonQuery();
    }

    // ── Write ──────────────────────────────────────────────────────────────

    public async Task<long> AddDocumentAsync(string filename, List<TextChunk> chunks, List<float[]> embeddings)
    {
        using var conn = Open();
        using var tx   = conn.BeginTransaction();

        long docId;
        using (var cmd = conn.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = """
                INSERT INTO documents (filename, uploaded_at, chunk_count)
                VALUES (@filename, @uploaded_at, @chunk_count);
                SELECT last_insert_rowid();
                """;
            cmd.Parameters.AddWithValue("@filename",    filename);
            cmd.Parameters.AddWithValue("@uploaded_at", DateTime.UtcNow.ToString("O"));
            cmd.Parameters.AddWithValue("@chunk_count", chunks.Count);
            docId = (long)(await cmd.ExecuteScalarAsync())!;
        }

        for (var i = 0; i < chunks.Count; i++)
        {
            long chunkId;
            using (var cmd = conn.CreateCommand())
            {
                cmd.Transaction = tx;
                cmd.CommandText = """
                    INSERT INTO chunks (document_id, page_number, chunk_index, text, embedding, y_start, y_end)
                    VALUES (@doc_id, @page_number, @chunk_index, @text, @embedding, @y_start, @y_end);
                    SELECT last_insert_rowid();
                    """;
                cmd.Parameters.AddWithValue("@doc_id",      docId);
                cmd.Parameters.AddWithValue("@page_number", chunks[i].PageNumber);
                cmd.Parameters.AddWithValue("@chunk_index", chunks[i].ChunkIndex);
                cmd.Parameters.AddWithValue("@text",        chunks[i].Text);
                cmd.Parameters.AddWithValue("@embedding",   FloatsToBytes(embeddings[i]));
                cmd.Parameters.AddWithValue("@y_start",     chunks[i].YStart);
                cmd.Parameters.AddWithValue("@y_end",       chunks[i].YEnd);
                chunkId = (long)(await cmd.ExecuteScalarAsync())!;
            }

            // Populate FTS index
            using (var cmd = conn.CreateCommand())
            {
                cmd.Transaction = tx;
                cmd.CommandText = "INSERT INTO chunks_fts(rowid, text) VALUES (@rowid, @text)";
                cmd.Parameters.AddWithValue("@rowid", chunkId);
                cmd.Parameters.AddWithValue("@text",  chunks[i].Text);
                await cmd.ExecuteNonQueryAsync();
            }
        }

        tx.Commit();
        _logger.LogInformation("Stored {File} — {N} chunks (id={Id})", filename, chunks.Count, docId);
        return docId;
    }

    public async Task DeleteDocumentAsync(long id)
    {
        // Collect chunk IDs for FTS cleanup first
        var chunkIds = new List<long>();
        using (var conn = Open())
        {
            using var cmd = conn.CreateCommand();
            cmd.CommandText = "SELECT id FROM chunks WHERE document_id = @id";
            cmd.Parameters.AddWithValue("@id", id);
            await using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync()) chunkIds.Add(reader.GetInt64(0));
        }

        using var wconn = Open();
        using var tx    = wconn.BeginTransaction();

        // Remove FTS entries
        foreach (var chunkId in chunkIds)
        {
            using var cmd = wconn.CreateCommand();
            cmd.Transaction = tx;
            cmd.CommandText = "DELETE FROM chunks_fts WHERE rowid = @rowid";
            cmd.Parameters.AddWithValue("@rowid", chunkId);
            await cmd.ExecuteNonQueryAsync();
        }

        using (var cmd = wconn.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "DELETE FROM chunks WHERE document_id = @id";
            cmd.Parameters.AddWithValue("@id", id);
            await cmd.ExecuteNonQueryAsync();
        }

        using (var cmd = wconn.CreateCommand())
        {
            cmd.Transaction = tx;
            cmd.CommandText = "DELETE FROM documents WHERE id = @id";
            cmd.Parameters.AddWithValue("@id", id);
            await cmd.ExecuteNonQueryAsync();
        }

        tx.Commit();

        // Delete stored PDF file
        var pdfPath = GetPdfPath(id);
        if (File.Exists(pdfPath))
            try { File.Delete(pdfPath); } catch { /* best-effort */ }

        _logger.LogInformation("Deleted document id={Id}", id);
    }

    // ── PDF file storage ───────────────────────────────────────────────────

    public async Task SavePdfAsync(long docId, byte[] bytes)
    {
        Directory.CreateDirectory(_pdfDir);
        await File.WriteAllBytesAsync(GetPdfPath(docId), bytes);
    }

    public string GetPdfPath(long docId) => Path.Combine(_pdfDir, $"{docId}.pdf");

    // ── Read ───────────────────────────────────────────────────────────────

    public async Task<List<DocumentInfo>> ListDocumentsAsync()
    {
        using var conn = Open();
        using var cmd  = conn.CreateCommand();
        cmd.CommandText = "SELECT id, filename, uploaded_at, chunk_count FROM documents ORDER BY id DESC";

        var results = new List<DocumentInfo>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            results.Add(new DocumentInfo(
                reader.GetInt64(0),
                reader.GetString(1),
                reader.GetString(2),
                reader.GetInt32(3)));

        return results;
    }

    /// <summary>
    /// Full-text search via SQLite FTS5 with BM25 ranking.
    /// Falls back to an empty list if the query is empty.
    /// </summary>
    public async Task<List<CandidateChunk>> SearchByKeywordsAsync(string[] terms, string[] variants)
    {
        var ftsQuery = BuildFtsQuery(terms, variants);
        if (string.IsNullOrWhiteSpace(ftsQuery)) return [];

        using var conn = Open();
        using var cmd  = conn.CreateCommand();
        cmd.CommandText = """
            SELECT c.document_id, d.filename, c.page_number, c.chunk_index,
                   c.text, c.embedding, c.y_start, c.y_end
            FROM chunks_fts f
            JOIN chunks    c ON c.id = f.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH @query
            ORDER BY rank
            LIMIT 200
            """;
        cmd.Parameters.AddWithValue("@query", ftsQuery);

        return await ReadCandidates(cmd);
    }

    public async Task<List<CandidateChunk>> GetAllChunkSampleAsync(int n)
    {
        using var conn = Open();
        using var cmd  = conn.CreateCommand();
        cmd.CommandText = """
            SELECT c.document_id, d.filename, c.page_number, c.chunk_index,
                   c.text, c.embedding, c.y_start, c.y_end
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY RANDOM()
            LIMIT @n
            """;
        cmd.Parameters.AddWithValue("@n", n);
        return await ReadCandidates(cmd);
    }

    // ── Rerank ─────────────────────────────────────────────────────────────

    public List<RetrievedChunk> RerankByEmbedding(float[] queryEmbedding, List<CandidateChunk> candidates, int topK)
    {
        return candidates
            .Select(c => (
                Result: new RetrievedChunk(c.DocumentId, c.DocumentName, c.PageNumber, c.ChunkIndex, c.Text, c.YStart, c.YEnd),
                Score:  CosineSimilarity(queryEmbedding, c.Embedding)))
            .OrderByDescending(x => x.Score)
            .Take(topK)
            .Select(x => x.Result)
            .ToList();
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static string BuildFtsQuery(string[] terms, string[] variants)
    {
        var all = terms.Concat(variants)
            .Where(t => !string.IsNullOrWhiteSpace(t))
            .Select(t => t.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (all.Length == 0) return "";

        // Always double-quote every term so FTS5 never interprets content as column
        // references, operators (OR/NOT/AND), or other special syntax.
        // "phrase with spaces" → exact phrase match
        // "singleword"*        → prefix match (quotes + * is valid FTS5 syntax)
        var parts = all.Select(t =>
        {
            var escaped = t.Replace("\"", "\"\""); // escape embedded quotes
            return t.Contains(' ')
                ? $"\"{escaped}\""    // exact phrase
                : $"\"{escaped}\"*";  // prefix match
        });

        return string.Join(" OR ", parts);
    }

    private static async Task<List<CandidateChunk>> ReadCandidates(SqliteCommand cmd)
    {
        var results = new List<CandidateChunk>();
        await using var reader = await cmd.ExecuteReaderAsync();
        while (await reader.ReadAsync())
            results.Add(new CandidateChunk(
                reader.GetInt64(0),
                reader.GetString(1),
                reader.GetInt32(2),
                reader.GetInt32(3),
                reader.GetString(4),
                BytesToFloats((byte[])reader.GetValue(5)),
                reader.GetFloat(6),
                reader.GetFloat(7)));
        return results;
    }

    private SqliteConnection Open()
    {
        var conn = new SqliteConnection($"Data Source={_dbPath}");
        conn.Open();
        return conn;
    }

    private static byte[] FloatsToBytes(float[] floats)
    {
        var bytes = new byte[floats.Length * sizeof(float)];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] BytesToFloats(byte[] bytes)
    {
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length) return 0f;
        float dot = 0f, magA = 0f, magB = 0f;
        for (var i = 0; i < a.Length; i++) { dot += a[i] * b[i]; magA += a[i] * a[i]; magB += b[i] * b[i]; }
        var denom = MathF.Sqrt(magA) * MathF.Sqrt(magB);
        return denom == 0f ? 0f : dot / denom;
    }
}
