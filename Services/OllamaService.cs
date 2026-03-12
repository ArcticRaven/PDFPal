using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace PdfPal.Services;

public class OllamaService
{
    private readonly HttpClient _http;
    private readonly string _chatModelGpu;
    private readonly string _chatModelCpu;
    private readonly string _embedModel;
    private readonly ILogger<OllamaService> _logger;
    private bool _hasGpu;

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    public string ChatModelGpu    => _chatModelGpu;
    public string ChatModelCpu    => _chatModelCpu;
    public string EmbedModel      => _embedModel;
    public string ActiveChatModel => _hasGpu ? _chatModelGpu : _chatModelCpu;

    public OllamaService(HttpClient http, IConfiguration config, ILogger<OllamaService> logger)
    {
        _http         = http;
        _chatModelGpu = config["Ollama:ChatModelGpu"] ?? "llama3.1:8b";
        _chatModelCpu = config["Ollama:ChatModelCpu"] ?? "llama3.2:1b";
        _embedModel   = config["Ollama:EmbedModel"]   ?? "nomic-embed-text";
        _logger       = logger;

        // Explicit override via Ollama:UseGpu — bypasses the unreliable auto-detection.
        // Set Ollama__UseGpu=true in docker-compose when running with an NVIDIA GPU.
        if (config.GetValue<bool?>("Ollama:UseGpu") is bool explicitGpu)
            _hasGpu = explicitGpu;
    }

    public void SetGpuMode(bool hasGpu) => _hasGpu = hasGpu;

    /// <summary>Single embedding — used for query vectors at retrieval time.</summary>
    public async Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default)
    {
        var payload  = new { model = _embedModel, input = text, keep_alive = "30m" };
        var response = await _http.PostAsJsonAsync("/api/embed", payload, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<BatchEmbedResponse>(_jsonOptions, cancellationToken: ct);
        return result?.Embeddings?.FirstOrDefault() ?? [];
    }

    /// <summary>
    /// Batch embedding — sends texts in one call per batch.
    /// On 400 (chunk too long / batch too large) automatically halves the batch and retries,
    /// ultimately falling back to one-at-a-time with truncation if needed.
    /// </summary>
    public async Task<List<float[]>> GetEmbeddingsBatchAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        if (texts.Count == 0) return [];
        return await EmbedWithFallbackAsync(texts, ct);
    }

    private async Task<List<float[]>> EmbedWithFallbackAsync(IReadOnlyList<string> texts, CancellationToken ct)
    {
        // Try the whole list as one request first
        try
        {
            var payload  = new { model = _embedModel, input = texts, keep_alive = "30m" };
            var response = await _http.PostAsJsonAsync("/api/embed", payload, ct);
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<BatchEmbedResponse>(_jsonOptions, cancellationToken: ct);
                if (result?.Embeddings is { Length: > 0 } vecs) return vecs.ToList();
            }
        }
        catch { /* fall through to smaller batches */ }

        // If the batch has more than one item, split and recurse
        if (texts.Count > 1)
        {
            _logger.LogDebug("Embed batch of {N} failed, splitting in half", texts.Count);
            var mid  = texts.Count / 2;
            var left  = await EmbedWithFallbackAsync(texts.Take(mid).ToList(), ct);
            var right = await EmbedWithFallbackAsync(texts.Skip(mid).ToList(), ct);
            left.AddRange(right);
            return left;
        }

        // Single chunk still failing — cascade through progressively smaller truncations.
        // Dense technical text (tables, measurements, formulas) tokenizes at 3-5 tokens/char,
        // so a 4000-char chunk can easily exceed a 2048-token context window.
        foreach (var limit in new[] { 3000, 2000, 1000, 400 })
        {
            var candidate = texts[0].Length > limit ? texts[0][..limit] : texts[0];
            _logger.LogWarning("Single chunk embed failed — retrying with {Limit}-char truncation ({Len} chars)", limit, candidate.Length);
            try
            {
                var payload  = new { model = _embedModel, input = candidate, keep_alive = "30m" };
                var response = await _http.PostAsJsonAsync("/api/embed", payload, ct);
                if (response.IsSuccessStatusCode)
                {
                    var result = await response.Content.ReadFromJsonAsync<BatchEmbedResponse>(_jsonOptions, cancellationToken: ct);
                    if (result?.Embeddings is { Length: > 0 }) return result.Embeddings.ToList();
                }
            }
            catch { /* try next limit */ }

            // If the text was already shorter than this limit, no point trying smaller — just bail
            if (texts[0].Length <= limit) break;
        }

        _logger.LogError("Embed failed at all truncation levels for chunk ({Len} chars) — using zero vector", texts[0].Length);
        return [new float[384]]; // nomic-embed-text dimension; prevents ingestion from aborting
    }

    public async IAsyncEnumerable<string> StreamChatAsync(
        string systemPrompt,
        string userMessage,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var payload = new
        {
            model      = ActiveChatModel,
            stream     = true,
            keep_alive = "30m",
            messages   = new[]
            {
                new { role = "system", content = systemPrompt },
                new { role = "user",   content = userMessage  }
            }
        };

        var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json")
        };

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader       = new StreamReader(stream);

        while (!reader.EndOfStream && !ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct);
            if (string.IsNullOrWhiteSpace(line)) continue;

            ChatStreamChunk? chunk = null;
            try { chunk = JsonSerializer.Deserialize<ChatStreamChunk>(line); }
            catch { /* skip malformed lines */ }

            if (chunk?.Message?.Content is { Length: > 0 } content)
                yield return content;

            if (chunk?.Done == true) break;
        }
    }

    public async Task<bool> IsHealthyAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _http.GetAsync("/api/tags", ct);
            return response.IsSuccessStatusCode;
        }
        catch { return false; }
    }

    public async Task<List<string>> ListModelsAsync(CancellationToken ct = default)
    {
        try
        {
            var response = await _http.GetFromJsonAsync<TagsResponse>("/api/tags", ct);
            return response?.Models?.Select(m => m.Name).ToList() ?? [];
        }
        catch { return []; }
    }

    /// <summary>
    /// Pulls a model from Ollama using the streaming API so the connection stays alive
    /// regardless of how long the download takes (avoids HttpClient timeout on large models).
    /// Logs progress as it arrives.
    /// </summary>
    public async Task PullModelAsync(string model, CancellationToken ct = default)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/pull")
        {
            Content = new StringContent(
                JsonSerializer.Serialize(new { name = model, stream = true }),
                Encoding.UTF8, "application/json")
        };

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader       = new StreamReader(stream);

        while (!reader.EndOfStream && !ct.IsCancellationRequested)
        {
            var line = await reader.ReadLineAsync(ct);
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                using var doc = JsonDocument.Parse(line);
                if (doc.RootElement.TryGetProperty("status", out var status))
                    _logger.LogInformation("[pull {Model}] {Status}", model, status.GetString());
                // "success" in the status field means the pull is complete
                if (doc.RootElement.TryGetProperty("status", out var s) &&
                    s.GetString() == "success") break;
            }
            catch { /* skip malformed lines */ }
        }
    }

    /// <summary>
    /// Attempts GPU detection by loading a tiny model and checking reported GPU layers.
    /// Falls back to false (CPU mode) on any failure.
    /// Prefer setting Ollama__UseGpu explicitly in docker-compose instead.
    /// </summary>
    public async Task<bool> HasGpuAsync(CancellationToken ct = default)
    {
        try
        {
            // Ask Ollama to load the embed model (small) and inspect how many layers went to GPU.
            var payload  = new { model = _embedModel, prompt = " ", stream = false };
            var response = await _http.PostAsJsonAsync("/api/generate", payload, ct);
            if (!response.IsSuccessStatusCode) return false;

            using var doc = await JsonDocument.ParseAsync(
                await response.Content.ReadAsStreamAsync(ct), cancellationToken: ct);

            // Ollama reports eval_count only when the model actually ran.
            // Check the model_info block for gpu layers if present.
            if (doc.RootElement.TryGetProperty("model_info", out var info))
            {
                foreach (var p in info.EnumerateObject())
                    if (p.Name.Contains("gpu", StringComparison.OrdinalIgnoreCase) &&
                        p.Value.TryGetInt64(out var n) && n > 0)
                        return true;
            }
            return false;
        }
        catch { return false; }
    }

    // ── DTOs ───────────────────────────────────────────────────────────────

    private sealed record BatchEmbedResponse(
        [property: JsonPropertyName("embeddings")] float[][] Embeddings);

    private sealed record ChatStreamChunk(
        [property: JsonPropertyName("message")]  ChatMessage? Message,
        [property: JsonPropertyName("done")]     bool Done);

    private sealed record ChatMessage(
        [property: JsonPropertyName("role")]    string Role,
        [property: JsonPropertyName("content")] string Content);

    private sealed record TagsResponse(
        [property: JsonPropertyName("models")] List<ModelInfo>? Models);

    private sealed record ModelInfo(
        [property: JsonPropertyName("name")] string Name);
}

// ── Public result type (used by EmbeddingService) ──────────────────────────

public sealed record KeywordResult(
    [property: JsonPropertyName("terms")]    string[] Terms,
    [property: JsonPropertyName("variants")] string[] Variants);
