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

    public string ActiveChatModel => _hasGpu ? _chatModelGpu : _chatModelCpu;

    public OllamaService(HttpClient http, IConfiguration config, ILogger<OllamaService> logger)
    {
        _http         = http;
        _chatModelGpu = config["Ollama:ChatModelGpu"] ?? "llama3.1:8b";
        _chatModelCpu = config["Ollama:ChatModelCpu"] ?? "llama3.2:1b";
        _embedModel   = config["Ollama:EmbedModel"]   ?? "nomic-embed-text";
        _logger       = logger;
    }

    public void SetGpuMode(bool hasGpu) => _hasGpu = hasGpu;

    public async Task<float[]> GetEmbeddingAsync(string text, CancellationToken ct = default)
    {
        var payload  = new { model = _embedModel, prompt = text };
        var response = await _http.PostAsJsonAsync("/api/embeddings", payload, ct);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<EmbeddingResponse>(cancellationToken: ct);
        return result?.Embedding ?? [];
    }

    public async IAsyncEnumerable<string> StreamChatAsync(
        string systemPrompt,
        string userMessage,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        var payload = new
        {
            model    = ActiveChatModel,
            stream   = true,
            messages = new[]
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

    /// <summary>
    /// Non-streaming call that asks the model to extract specific keyword terms from a query,
    /// taking into account the current chat history and loaded document names.
    /// Falls back gracefully to empty arrays on any failure.
    /// </summary>
    public async Task<KeywordResult> ExtractKeywordsAsync(
        string query,
        string chatHistory,
        string documentNames,
        CancellationToken ct = default)
    {
        var prompt = $$"""
                       You are helping search a document database.
                       Chat history: {{chatHistory}}
                       Documents loaded: {{documentNames}}
                       Current query: "{{query}}"

                       Extract specific, technical, or rare search terms from the query considering the context above. Ignore common words.
                       Return JSON only, no explanation: {"terms": ["term1", "term2"], "variants": ["variant1", "variant2"]}
                       """;

        var payload = new
        {
            model = ActiveChatModel,
            stream = false,
            messages = new[]
            {
                new { role = "user", content = prompt }
            }
        };

        try
        {
            var response = await _http.PostAsJsonAsync("/api/chat", payload, ct);
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadFromJsonAsync<NonStreamingChatResponse>(
                _jsonOptions, cancellationToken: ct);
            var content = result?.Message?.Content ?? "";

            var jsonStart = content.IndexOf('{');
            var jsonEnd = content.LastIndexOf('}');
            if (jsonStart >= 0 && jsonEnd > jsonStart)
            {
                var json = content[jsonStart..(jsonEnd + 1)];
                var kw = JsonSerializer.Deserialize<KeywordResult>(json, _jsonOptions);
                if (kw != null) return kw;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Keyword extraction failed — falling back to embedding-only search");
        }

        return new KeywordResult([], []);
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

    public async Task PullModelAsync(string model, CancellationToken ct = default)
    {
        var response = await _http.PostAsJsonAsync("/api/pull", new { name = model, stream = false }, ct);
        response.EnsureSuccessStatusCode();
    }

    public async Task<bool> HasGpuAsync(CancellationToken ct = default)
    {
        try
        {
            var body = await _http.GetStringAsync("/api/version", ct);
            return body.Contains("cuda", StringComparison.OrdinalIgnoreCase);
        }
        catch { return false; }
    }

    // ── DTOs ───────────────────────────────────────────────────────────────

    private sealed record EmbeddingResponse(
        [property: JsonPropertyName("embedding")] float[] Embedding);

    private sealed record ChatStreamChunk(
        [property: JsonPropertyName("message")]  ChatMessage? Message,
        [property: JsonPropertyName("done")]     bool Done);

    private sealed record NonStreamingChatResponse(
        [property: JsonPropertyName("message")] ChatMessage? Message);

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
