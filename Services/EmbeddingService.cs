namespace PdfPal.Services;

public class EmbeddingService
{
    private readonly OllamaService  _ollama;
    private readonly PdfService     _pdf;
    private readonly DocumentStore  _documentStore;
    private readonly ILogger<EmbeddingService> _logger;

    public EmbeddingService(
        OllamaService  ollama,
        PdfService     pdf,
        DocumentStore  documentStore,
        ILogger<EmbeddingService> logger)
    {
        _ollama        = ollama;
        _pdf           = pdf;
        _documentStore = documentStore;
        _logger        = logger;
    }

    /// <summary>
    /// Parse, tag, embed, and persist a PDF.
    /// NLP tagging runs in parallel (CPU-bound, thread-safe).
    /// Embedding uses batched /api/embed calls.
    /// </summary>
    public async Task IngestAsync(
        byte[]      pdfBytes,
        string      fileName,
        Action<int> onProgress,
        CancellationToken ct = default)
    {
        _logger.LogInformation("Ingesting {File} ({Bytes} bytes)", fileName, pdfBytes.Length);
        onProgress(5);

        // 1. Extract positioned chunks
        using var stream = new MemoryStream(pdfBytes);
        var chunks = _pdf.ExtractChunks(stream);
        _logger.LogInformation("Extracted {N} chunks from {File}", chunks.Count, fileName);

        if (chunks.Count == 0)
            throw new InvalidOperationException("No text could be extracted from this PDF.");

        onProgress(10);

        // 2. NLP tagging — pure CPU work, run in parallel across all chunks
        _logger.LogInformation("Tagging {N} chunks (parallel NLP)…", chunks.Count);
        var chunkTags = await Task.Run(() =>
            chunks
                .AsParallel()
                .AsOrdered()
                .Select(c => NlpTagger.TagChunk(c.Text))
                .ToList(), ct);

        onProgress(20);

        // 3. Embed chunks in batches (20 → 95 %)
        const int BatchSize = 32;
        var embeddings = new List<float[]>(chunks.Count);
        for (var i = 0; i < chunks.Count; i += BatchSize)
        {
            ct.ThrowIfCancellationRequested();
            var batch     = chunks.Skip(i).Take(BatchSize).Select(c => c.Text).ToList();
            var batchVecs = await _ollama.GetEmbeddingsBatchAsync(batch, ct);
            embeddings.AddRange(batchVecs);
            onProgress(20 + (int)(Math.Min(i + BatchSize, chunks.Count) / (float)chunks.Count * 75));
        }

        // 4. Persist chunks + tags + FTS index
        var docId = await _documentStore.AddDocumentAsync(fileName, chunks, embeddings, chunkTags);

        // 5. Save original PDF bytes so the viewer can serve them
        await _documentStore.SavePdfAsync(docId, pdfBytes);

        onProgress(100);
        _logger.LogInformation("Ingestion complete for {File} (docId={Id})", fileName, docId);
    }

    /// <summary>
    /// Three-mode hybrid retrieval using NLP query expansion (no LLM warm-up).
    /// </summary>
    public async Task<List<RetrievedChunk>> RetrieveAsync(
        string query,
        int    topK = 6,
        CancellationToken ct = default)
    {
        var (chunks, _, _) = await RetrieveWithDebugAsync(query, topK, ct);
        return chunks;
    }

    /// <summary>
    /// Same as RetrieveAsync but also returns keyword/tag info for the debug drawer.
    /// </summary>
    public async Task<(List<RetrievedChunk> Chunks, KeywordResult Keywords, int CandidateCount)> RetrieveWithDebugAsync(
        string query,
        int    topK = 6,
        CancellationToken ct = default)
    {
        // NLP-based query expansion — instant, no LLM call
        var keywords = NlpTagger.ExtractQueryTerms(query);

        List<CandidateChunk> candidates;

        if (keywords.Terms.Length > 0 || keywords.Variants.Length > 0)
        {
            // Mode 1 — FTS5 search across text + tags columns (BM25-ranked)
            candidates = await _documentStore.SearchByKeywordsAsync(keywords.Terms, keywords.Variants);

            // Mode 2 — sparse: supplement with a random sample when FTS hits are few
            if (candidates.Count < 10)
            {
                _logger.LogDebug("FTS sparse ({N} hits), expanding with random sample", candidates.Count);
                var sample       = await _documentStore.GetAllChunkSampleAsync(200);
                var existingKeys = candidates.Select(c => (c.DocumentId, c.ChunkIndex)).ToHashSet();
                candidates.AddRange(sample.Where(s => existingKeys.Add((s.DocumentId, s.ChunkIndex))));
            }
        }
        else
        {
            // Mode 3 — no extractable terms: full random sample
            _logger.LogDebug("No NLP terms extracted, using random sample");
            candidates = await _documentStore.GetAllChunkSampleAsync(200);
        }

        if (candidates.Count == 0)
            return ([], keywords, 0);

        var queryEmbedding = await _ollama.GetEmbeddingAsync(query, ct);
        var chunks         = _documentStore.RerankByEmbedding(queryEmbedding, candidates, topK);
        return (chunks, keywords, candidates.Count);
    }
}
