namespace PdfPal.Services;

/// <summary>
/// Singleton that tracks conversation history with a rolling window of up to 20 messages.
/// </summary>
public class ChatSession
{
    private readonly List<(string Role, string Content)> _history = [];
    private readonly object _lock = new();
    private const int MaxMessages = 20;

    public void Add(string role, string content)
    {
        lock (_lock)
        {
            _history.Add((role, content));
            // Rolling window — trim oldest if over limit
            while (_history.Count > MaxMessages)
                _history.RemoveAt(0);
        }
    }

    public List<(string Role, string Content)> GetRecent(int n)
    {
        lock (_lock)
        {
            var start = Math.Max(0, _history.Count - n);
            return _history.Skip(start).ToList();
        }
    }

    public void Clear()
    {
        lock (_lock) { _history.Clear(); }
    }

    /// <summary>
    /// Returns the last <paramref name="n"/> messages formatted as a plain string for prompt injection.
    /// </summary>
    public string FormatForPrompt(int n = 10)
    {
        var recent = GetRecent(n);
        if (recent.Count == 0) return "(no previous conversation)";
        return string.Join("\n", recent.Select(m => $"{m.Role}: {m.Content}"));
    }
}
