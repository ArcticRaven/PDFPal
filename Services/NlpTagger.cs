using System.Text.RegularExpressions;

namespace PdfPal.Services;

/// <summary>
/// Lightweight rule-based NLP tagger — no model or warm-up required.
///
/// Extracts four classes of terms from text:
///   1. Acronyms        — ISO, ASTM, UNF, SAE, NF, HRC …
///   2. Noun phrases    — consecutive Title-Case words: "Ball Bearing", "Tensile Strength"
///   3. Bigrams         — pairs of significant (non-stop) words: "thread pitch", "heat treatment"
///   4. Capitalized     — mid-sentence proper-noun-style words: "Rockwell", "Brinell"
///
/// Used both for indexing chunks at ingest time (stored in the `tags` column / FTS5)
/// and for expanding user queries at retrieval time (replaces the LLM keyword call).
/// </summary>
public static class NlpTagger
{
    // ── Stop words ─────────────────────────────────────────────────────────

    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "a","an","the","and","or","but","in","on","at","to","for","of","with","by",
        "from","is","are","was","were","be","been","have","has","had","do","does",
        "did","will","would","could","should","may","might","shall","can","this",
        "that","these","those","it","its","not","no","as","if","when","where",
        "which","who","what","how","all","each","every","both","more","most",
        "other","some","such","than","then","so","up","out","about","into",
        "through","during","before","after","above","below","between","any","also",
        "being","their","they","them","we","us","you","your","he","she","his",
        "her","our","my","i","me","am","per","see","used","use","using","based",
        "figure","table","section","chapter","page","note","shown","given",
        "value","values","number","numbers","following","however","therefore",
        "thus","since","where","example","generally","typically","usually"
    };

    // ── Pre-compiled regexes ───────────────────────────────────────────────

    // 2–6 uppercase letters (not followed by lowercase — avoids sentence-start caps)
    private static readonly Regex AcronymRe = new(
        @"\b[A-Z]{2,6}\b",
        RegexOptions.Compiled);

    // Two or more consecutive Title-Case words
    private static readonly Regex NounPhraseRe = new(
        @"\b[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+\b",
        RegexOptions.Compiled);

    // Any word 3+ letters (for bigram + stop-word filtering)
    private static readonly Regex WordRe = new(
        @"\b[a-zA-Z]{3,}\b",
        RegexOptions.Compiled);

    // Title-Case word appearing after a lowercase character (mid-sentence proper noun)
    private static readonly Regex MidSentenceCapRe = new(
        @"(?<=[a-z,;:(]\s)[A-Z][a-z]{2,}\b",
        RegexOptions.Compiled);

    // ── Public API ─────────────────────────────────────────────────────────

    /// <summary>
    /// Tags a chunk of text for FTS5 indexing.
    /// Returns a space-separated string stored in the `tags` column.
    /// Safe to call in parallel (stateless).
    /// </summary>
    public static string TagChunk(string text)
    {
        var tags = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // 1. Acronyms
        foreach (Match m in AcronymRe.Matches(text))
            tags.Add(m.Value);

        // 2. Noun phrases + constituent words
        foreach (Match m in NounPhraseRe.Matches(text))
        {
            tags.Add(m.Value);
            foreach (var word in m.Value.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                if (!StopWords.Contains(word))
                    tags.Add(word.ToLowerInvariant());
        }

        // 3. Bigrams from non-stop words
        var words = WordRe.Matches(text)
            .Cast<Match>()
            .Select(m => m.Value.ToLowerInvariant())
            .Where(w => !StopWords.Contains(w))
            .ToList();

        for (var i = 0; i < words.Count - 1; i++)
            tags.Add($"{words[i]} {words[i + 1]}");

        // 4. Mid-sentence capitalized words (proper nouns)
        foreach (Match m in MidSentenceCapRe.Matches(text))
            if (!StopWords.Contains(m.Value))
                tags.Add(m.Value.ToLowerInvariant());

        return string.Join(" ", tags.Where(t => t.Length >= 2));
    }

    /// <summary>
    /// Extracts search terms from a user query for FTS5 lookup.
    /// Instant — no LLM call. Returns the same KeywordResult shape the
    /// debug drawer and retrieval pipeline already expect.
    /// </summary>
    public static KeywordResult ExtractQueryTerms(string query)
    {
        var terms    = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var variants = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Acronyms → terms
        foreach (Match m in AcronymRe.Matches(query))
            terms.Add(m.Value);

        // Noun phrases → variants
        foreach (Match m in NounPhraseRe.Matches(query))
            variants.Add(m.Value);

        // Individual significant words → terms
        foreach (Match m in WordRe.Matches(query))
        {
            var w = m.Value.ToLowerInvariant();
            if (!StopWords.Contains(w))
                terms.Add(w);
        }

        // Bigrams → variants
        var words = WordRe.Matches(query)
            .Cast<Match>()
            .Select(m => m.Value.ToLowerInvariant())
            .Where(w => !StopWords.Contains(w))
            .ToList();

        for (var i = 0; i < words.Count - 1; i++)
            variants.Add($"{words[i]} {words[i + 1]}");

        return new KeywordResult(
            terms.ToArray(),
            variants.Where(v => !terms.Contains(v, StringComparer.OrdinalIgnoreCase)).ToArray());
    }
}
