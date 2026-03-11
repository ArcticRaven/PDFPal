using System.Collections.Concurrent;
using System.Text;
using PdfPal.Services;

var builder = WebApplication.CreateBuilder(args);

// ── Services ───────────────────────────────────────────────────────────────

builder.Services.AddSingleton<DocumentStore>();
builder.Services.AddSingleton<ChatSession>();
builder.Services.AddSingleton<PdfService>();
builder.Services.AddSingleton<EmbeddingService>();
builder.Services.AddSingleton<OllamaService>();
builder.Services.AddHostedService<OllamaStartupService>();

builder.Services.AddHttpClient<OllamaService>(client =>
{
    var ollamaUrl = builder.Configuration["Ollama:BaseUrl"] ?? "http://ollama:11434";
    client.BaseAddress = new Uri(ollamaUrl);
    client.Timeout = TimeSpan.FromMinutes(10);
});

builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
    p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader()));

var app = builder.Build();

app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();

// Wipe DB + PDFs on clean shutdown
app.Lifetime.ApplicationStopping.Register(() =>
{
    var store  = app.Services.GetRequiredService<DocumentStore>();
    var logger = app.Services.GetRequiredService<ILogger<Program>>();
    logger.LogInformation("Wiping data on shutdown…");
    store.Wipe();
});

// ── Status ─────────────────────────────────────────────────────────────────

app.MapGet("/api/status", async (OllamaService ollama, DocumentStore store) =>
{
    var healthy   = await ollama.IsHealthyAsync();
    var models    = healthy ? await ollama.ListModelsAsync() : [];
    var documents = await store.ListDocumentsAsync();

    return Results.Ok(new
    {
        ollamaReady   = healthy,
        models,
        documentCount = documents.Count,
        documents     = documents.Select(d => new { d.Id, d.Filename, d.ChunkCount })
    });
});

// ── Documents ──────────────────────────────────────────────────────────────

app.MapGet("/api/documents", async (DocumentStore store) =>
{
    var docs = await store.ListDocumentsAsync();
    return Results.Ok(docs.Select(d => new { d.Id, d.Filename, d.UploadedAt, d.ChunkCount }));
});

app.MapDelete("/api/documents/{id:long}", async (long id, DocumentStore store) =>
{
    await store.DeleteDocumentAsync(id);
    return Results.Ok(new { message = $"Document {id} deleted" });
});

// ── PDF file serving (for viewer) ──────────────────────────────────────────

app.MapGet("/api/documents/{id:long}/file", async (long id, DocumentStore store) =>
{
    var path = store.GetPdfPath(id);
    if (!File.Exists(path)) return Results.NotFound();
    var bytes = await File.ReadAllBytesAsync(path);
    return Results.File(bytes, "application/pdf");
});

// ── New session ────────────────────────────────────────────────────────────

app.MapPost("/api/session/new", (DocumentStore store, ChatSession session) =>
{
    store.Reset();
    session.Clear();
    return Results.Ok(new { message = "New session started" });
});

// ── Upload & ingest ────────────────────────────────────────────────────────

var ingestionProgress = new ConcurrentDictionary<string, IngestionState>();

app.MapPost("/api/upload", async (
    HttpRequest      request,
    EmbeddingService embedding,
    ILogger<Program> logger) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest("Expected multipart/form-data");

    var form = await request.ReadFormAsync();
    var file = form.Files.FirstOrDefault();
    if (file is null)
        return Results.BadRequest("No file provided");

    if (!file.FileName.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
        return Results.BadRequest("Only PDF files are supported");

    var state = ingestionProgress.AddOrUpdate(
        file.FileName,
        _ => new IngestionState(file.FileName),
        (_, _) => new IngestionState(file.FileName));

    // Read entire upload into memory once so we can both ingest and save the bytes
    using var ms = new MemoryStream();
    await file.CopyToAsync(ms);
    var pdfBytes = ms.ToArray();
    var fileName = file.FileName;

    _ = Task.Run(async () =>
    {
        try
        {
            await embedding.IngestAsync(
                pdfBytes,
                fileName,
                progress => state.Update(progress),
                CancellationToken.None);
            state.Complete();
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Ingestion failed for {File}", fileName);
            state.Fail(ex.Message);
        }
    });

    return Results.Accepted("/api/progress", new { message = "Ingestion started", fileName });
});

app.MapGet("/api/progress", (string? file) =>
{
    if (file is not null && ingestionProgress.TryGetValue(file, out var state))
        return Results.Ok(state.Snapshot());

    return Results.Ok(new { FileName = file, Progress = 0, Status = "idle", Error = (string?)null });
});

// ── Chat ───────────────────────────────────────────────────────────────────

app.MapPost("/api/chat", async (
    ChatRequest      req,
    EmbeddingService embedding,
    OllamaService    ollama,
    DocumentStore    documentStore,
    ChatSession      chatSession,
    HttpContext      ctx) =>
{
    var documents = await documentStore.ListDocumentsAsync();
    if (documents.Count == 0)
        return Results.BadRequest("No documents loaded. Please upload a PDF first.");

    if (string.IsNullOrWhiteSpace(req.Message))
        return Results.BadRequest("Message cannot be empty.");

    // Retrieve with debug info
    var (chunks, keywords, candidateCount) = await embedding.RetrieveWithDebugAsync(req.Message, topK: 6);

    // Number excerpts for (1)(2) inline citations
    var contextBuilder = new StringBuilder();
    for (var i = 0; i < chunks.Count; i++)
        contextBuilder.AppendLine($"[{i + 1}] {chunks[i].DocumentName} — Page {chunks[i].PageNumber}\n{chunks[i].Text}\n");

    var systemPrompt = $"""
You are a precise document assistant. You answer questions strictly based on the provided document excerpts.

Relevant excerpts:
---
{contextBuilder}
---

Rules:
- Answer only from the excerpts above. Do not use outside knowledge.
- Cite sources inline using their number in parentheses, e.g. "The shutter speed dial (1) controls exposure."
- You may cite multiple sources for a single claim, e.g. (1)(3).
- If the answer is not found in the excerpts, say "I couldn't find that in the documents."
- Be concise and precise. Do not repeat the document name or page number in your prose — the citations handle that.
""";

    ctx.Response.ContentType = "text/event-stream";
    ctx.Response.Headers.CacheControl = "no-cache";
    ctx.Response.Headers.Connection   = "keep-alive";

    // ── debug event (shown in UI debug drawer) ─────────────────────────────
    var debugPayload = new
    {
        type           = "debug",
        keywords       = keywords.Terms,
        variants       = keywords.Variants,
        candidateCount,
        topChunks      = chunks.Select((c, i) => new
        {
            index    = i + 1,
            doc      = c.DocumentName,
            page     = c.PageNumber,
            yStart   = c.YStart,
            yEnd     = c.YEnd
        })
    };
    await ctx.Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(debugPayload)}\n\n");
    await ctx.Response.Body.FlushAsync();

    // ── meta event (source list for annotations) ───────────────────────────
    var sources = chunks.Select((c, i) => new
    {
        index  = i + 1,
        docId  = c.DocumentId,
        doc    = c.DocumentName,
        page   = c.PageNumber,
        yStart = c.YStart,
        yEnd   = c.YEnd
    }).ToList();

    await ctx.Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(new { type = "meta", sources })}\n\n");
    await ctx.Response.Body.FlushAsync();

    // ── stream tokens ──────────────────────────────────────────────────────
    var fullResponse = new StringBuilder();
    await foreach (var token in ollama.StreamChatAsync(systemPrompt, req.Message, ctx.RequestAborted))
    {
        fullResponse.Append(token);
        await ctx.Response.WriteAsync($"data: {System.Text.Json.JsonSerializer.Serialize(new { type = "token", value = token })}\n\n");
        await ctx.Response.Body.FlushAsync();
    }

    await ctx.Response.WriteAsync("data: {\"type\":\"done\"}\n\n");
    await ctx.Response.Body.FlushAsync();

    chatSession.Add("user",      req.Message);
    chatSession.Add("assistant", fullResponse.ToString());

    return Results.Empty;
});

app.Run();

// ── Supporting types ───────────────────────────────────────────────────────

record ChatRequest(string Message);

class IngestionState
{
    private int     _progress;
    private string  _status = "processing";
    private string? _error;
    public  string  FileName { get; }

    public IngestionState(string fileName) => FileName = fileName;
    public void Update(int p)      => Interlocked.Exchange(ref _progress, p);
    public void Complete()         { _progress = 100; _status = "done"; }
    public void Fail(string error) { _status = "error"; _error = error; }
    public object Snapshot() => new { FileName, Progress = _progress, Status = _status, Error = _error };
}

class OllamaStartupService : IHostedService
{
    private readonly IServiceProvider _services;
    private readonly IConfiguration   _config;
    private readonly ILogger<OllamaStartupService> _logger;

    public OllamaStartupService(IServiceProvider services, IConfiguration config, ILogger<OllamaStartupService> logger)
    {
        _services = services;
        _config   = config;
        _logger   = logger;
    }

    public async Task StartAsync(CancellationToken ct)
    {
        var ollama = _services.GetRequiredService<OllamaService>();

        _logger.LogInformation("Waiting for Ollama…");
        while (!ct.IsCancellationRequested)
        {
            if (await ollama.IsHealthyAsync(ct)) break;
            await Task.Delay(2000, ct);
        }

        var hasGpu = await ollama.HasGpuAsync(ct);
        ollama.SetGpuMode(hasGpu);
        _logger.LogInformation("GPU={HasGpu} — chat model: {Model}", hasGpu, ollama.ActiveChatModel);

        var existing = await ollama.ListModelsAsync(ct);
        await EnsureModelAsync(ollama, ollama.ActiveChatModel, existing, ct);
        await EnsureModelAsync(ollama, _config["Ollama:EmbedModel"] ?? "nomic-embed-text", existing, ct);
        _logger.LogInformation("All models ready.");
    }

    public Task StopAsync(CancellationToken ct) => Task.CompletedTask;

    private async Task EnsureModelAsync(OllamaService ollama, string model, List<string> existing, CancellationToken ct)
    {
        if (existing.Any(m => m.StartsWith(model))) { _logger.LogInformation("{Model} already present.", model); return; }
        _logger.LogInformation("Pulling {Model}…", model);
        await ollama.PullModelAsync(model, ct);
        _logger.LogInformation("{Model} ready.", model);
    }
}
