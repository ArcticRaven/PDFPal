namespace PdfPal.Services;

public class EmbeddingService
{
    private readonly OllamaService  _ollama;
    private readonly PdfService     _pdf;
    private readonly DocumentStore  _documentStore;
    private readonly ChatSession    _chatSession;
    private readonly ILogger<EmbeddingService> _logger;

    public EmbeddingService(
        OllamaService  ollama,
        PdfService     pdf,
        DocumentStore  documentStore,
        ChatSession    chatSession,
        ILogger<EmbeddingService> logger)
    {
        _ollama        = ollama;
        _pdf           = pdf;
        _documentStore = documentStore;
        _chatSession   = chatSession;
        _logger        = logger;
    }

    /// <summary>
    /// Parse, embed, and persist a PDF. Saves the original bytes so the viewer can serve it.
    /// Reports progress 0–100 via the callback.
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

        // 2. Embed each chunk (5 → 95 %)
        var embeddings = new List<float[]>(chunks.Count);
        for (var i = 0; i < chunks.Count; i++)
        {
            ct.ThrowIfCancellationRequested();
            embeddings.Add(await _ollama.GetEmbeddingAsync(chunks[i].Text, ct));
            onProgress(5 + (int)((i + 1) / (float)chunks.Count * 90));
        }

        // 3. Persist chunks + FTS index
        var docId = await _documentStore.AddDocumentAsync(fileName, chunks, embeddings);

        // 4. Save original PDF bytes so the viewer can serve them
        await _documentStore.SavePdfAsync(docId, pdfBytes);

        onProgress(100);
        _logger.LogInformation("Ingestion complete for {File} (docId={Id})", fileName, docId);
    }

    /// <summary>
    /// Three-mode hybrid retrieval: FTS keyword search → cosine rerank.
    /// Returns chunks with document ID and Y coordinates for the viewer.
    /// </summary>
    public async Task<List<RetrievedChunk>> RetrieveAsync(
        string query,
        int    topK = 6,
        CancellationToken ct = default)
    {
        var documents     = await _documentStore.ListDocumentsAsync();
        var documentNames = string.Join(", ", documents.Select(d => d.Filename));
        var chatHistory   = _chatSession.FormatForPrompt(10);

        var keywords = await _ollama.ExtractKeywordsAsync(query, chatHistory, documentNames, ct);

        List<CandidateChunk> candidates;

        if (keywords.Terms.Length > 0)
        {
            // Mode 1 — FTS keyword search (BM25-ranked)
            candidates = await _documentStore.SearchByKeywordsAsync(keywords.Terms, keywords.Variants);

            // Mode 2 — sparse: supplement with random sample
            if (candidates.Count < 10)
            {
                _logger.LogDebug("FTS sparse ({N} hits), expanding with random sample", candidates.Count);
                var sample      = await _documentStore.GetAllChunkSampleAsync(200);
                var existingKeys = candidates.Select(c => (c.DocumentId, c.ChunkIndex)).ToHashSet();
                candidates.AddRange(sample.Where(s => existingKeys.Add((s.DocumentId, s.ChunkIndex))));
            }
        }
        else
        {
            // Mode 3 — no keywords: full random sample
            _logger.LogDebug("No keywords extracted, using random sample");
            candidates = await _documentStore.GetAllChunkSampleAsync(200);
        }

        if (candidates.Count == 0) return [];

        var queryEmbedding = await _ollama.GetEmbeddingAsync(query, ct);
        return _documentStore.RerankByEmbedding(queryEmbedding, candidates, topK);
    }

    /// <summary>
    /// Same as RetrieveAsync but also returns the keyword extraction result for debug output.
    /// </summary>
    public async Task<(List<RetrievedChunk> Chunks, KeywordResult Keywords, int CandidateCount)> RetrieveWithDebugAsync(
        string query,
        int    topK = 6,
        CancellationToken ct = default)
    {
        var documents     = await _documentStore.ListDocumentsAsync();
        var documentNames = string.Join(", ", documents.Select(d => d.Filename));
        var chatHistory   = _chatSession.FormatForPrompt(10);

        var keywords = await _ollama.ExtractKeywordsAsync(query, chatHistory, documentNames, ct);

        List<CandidateChunk> candidates;

        if (keywords.Terms.Length > 0)
        {
            candidates = await _documentStore.SearchByKeywordsAsync(keywords.Terms, keywords.Variants);
            if (candidates.Count < 10)
            {
                var sample       = await _documentStore.GetAllChunkSampleAsync(200);
                var existingKeys = candidates.Select(c => (c.DocumentId, c.ChunkIndex)).ToHashSet();
                candidates.AddRange(sample.Where(s => existingKeys.Add((s.DocumentId, s.ChunkIndex))));
            }
        }
        else
        {
            candidates = await _documentStore.GetAllChunkSampleAsync(200);
        }

        if (candidates.Count == 0)
            return ([], keywords, 0);

        var queryEmbedding = await _ollama.GetEmbeddingAsync(query, ct);
        var chunks         = _documentStore.RerankByEmbedding(queryEmbedding, candidates, topK);
        return (chunks, keywords, candidates.Count);
    }
}
