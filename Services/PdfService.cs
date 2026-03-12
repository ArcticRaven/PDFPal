using iText.Kernel.Geom;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Data;
using iText.Kernel.Pdf.Canvas.Parser.Listener;

namespace PdfPal.Services;

public sealed record TextChunk(int PageNumber, int ChunkIndex, string Text, float YStart, float YEnd);

public class PdfService
{
    // Maximum words before a chunk is forcibly split.
    // Kept low because technical text (tables, measurements, formulas) tokenizes at
    // 2-3 tokens per character, so 500 words can easily exceed nomic-embed-text's
    // 8192-token context window. 200 words ≈ ~1200 chars ≈ safe for any model.
    private const int MaxWords  = 200;
    // Hard character cap as a safety net for very short but dense words (e.g. tables)
    private const int MaxChars  = 1500;
    // Don't emit near-empty chunks (e.g. lone headers)
    private const int MinWords  = 10;

    public List<TextChunk> ExtractChunks(Stream pdfStream)
    {
        var result = new List<TextChunk>();

        using var reader = new PdfReader(pdfStream);
        using var doc    = new PdfDocument(reader);

        for (var pageNum = 1; pageNum <= doc.GetNumberOfPages(); pageNum++)
        {
            var page       = doc.GetPage(pageNum);
            var pageHeight = page.GetPageSize().GetHeight();

            var collector = new LineCollector();
            try { new PdfCanvasProcessor(collector).ProcessPageContent(page); }
            catch { continue; } // skip unreadable pages

            if (collector.Lines.Count == 0) continue;

            result.AddRange(ChunkLines(collector.Lines, pageNum, pageHeight));
        }

        return result;
    }

    // ── Paragraph-aware chunking ───────────────────────────────────────────

    private static List<TextChunk> ChunkLines(List<TextLine> lines, int pageNum, float pageHeight)
    {
        var result     = new List<TextChunk>();
        var words      = new List<string>();
        var yTopPdf    = lines[0].YTop;     // highest PDF-Y seen so far (= visually topmost)
        var yBottomPdf = lines[0].YBottom;  // lowest PDF-Y seen so far  (= visually bottommost)
        var index      = 0;

        void Flush()
        {
            if (words.Count < MinWords) { words.Clear(); return; }

            // PDF Y=0 is at bottom; convert to top-down screen fractions
            var yStart = Math.Clamp(1f - yTopPdf    / pageHeight, 0f, 1f);
            var yEnd   = Math.Clamp(1f - yBottomPdf / pageHeight, 0f, 1f);
            if (yStart > yEnd) (yStart, yEnd) = (yEnd, yStart);

            result.Add(new TextChunk(pageNum, index++, string.Join(" ", words), yStart, yEnd));
            words.Clear();
        }

        for (var i = 0; i < lines.Count; i++)
        {
            var line      = lines[i];
            var lineWords = line.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // Paragraph break: the gap between this line and the previous exceeds 1.5× line height
            var paragraphBreak = i > 0 &&
                (lines[i - 1].YBottom - line.YTop) > line.Height * 1.5f;

            var currentChars = words.Sum(w => w.Length + 1);
            if (paragraphBreak && words.Count >= MaxWords / 2)              Flush();
            if (words.Count + lineWords.Length > MaxWords)                  Flush();
            if (currentChars + line.Text.Length > MaxChars && words.Count > 0) Flush();

            if (words.Count == 0)
            {
                // Starting a new chunk — record its top edge
                yTopPdf    = line.YTop;
                yBottomPdf = line.YBottom;
            }
            else
            {
                // Extend bottom edge as we add more lines
                yBottomPdf = line.YBottom;
            }

            words.AddRange(lineWords);
        }

        Flush();
        return result;
    }
}

// ── Line model ─────────────────────────────────────────────────────────────

/// <summary>
/// A single visual line of text with its Y bounds in PDF coordinate space (Y=0 at bottom).
/// </summary>
public sealed record TextLine(string Text, float YTop, float YBottom, float Height);

// ── iText7 event listener ──────────────────────────────────────────────────

/// <summary>
/// Collects text render events from iText7 and groups them into visual lines
/// by bucketing on Y coordinate.
/// </summary>
public sealed class LineCollector : IEventListener
{
    // Y bucket key → ordered list of (x, text) fragments
    private readonly Dictionary<int, List<(float X, string Text)>> _frags   = new();
    // Y bucket key → (topY, bottomY) bounds in PDF space
    private readonly Dictionary<int, (float Top, float Bottom)>    _yBounds = new();

    public void EventOccurred(IEventData data, EventType type)
    {
        if (data is not TextRenderInfo tri) return;

        var text = tri.GetText();
        if (string.IsNullOrEmpty(text)) return;

        var baseline  = tri.GetBaseline();
        var ascent    = tri.GetAscentLine();
        var yBaseline = baseline.GetStartPoint().Get(Vector.I2);
        var yAscent   = ascent.GetStartPoint().Get(Vector.I2);
        var x         = baseline.GetStartPoint().Get(Vector.I1);

        // Round to nearest 2pt to group text on the same visual line
        var bucket = (int)MathF.Round(yBaseline / 2f) * 2;

        if (!_frags.TryGetValue(bucket, out var frags))
        {
            frags = [];
            _frags[bucket]   = frags;
            _yBounds[bucket] = (yAscent, yBaseline);
        }
        else
        {
            var (top, bot)   = _yBounds[bucket];
            _yBounds[bucket] = (MathF.Max(top, yAscent), MathF.Min(bot, yBaseline));
        }

        frags.Add((x, text));
    }

    public ICollection<EventType> GetSupportedEvents() => new[] { EventType.RENDER_TEXT };

    /// <summary>
    /// Returns lines sorted top-to-bottom (descending PDF Y).
    /// </summary>
    public List<TextLine> Lines
    {
        get
        {
            var lines = new List<TextLine>();
            foreach (var bucket in _frags.Keys.OrderByDescending(k => k))
            {
                var frags  = _frags[bucket];
                var text   = string.Concat(frags.OrderBy(f => f.X).Select(f => f.Text)).Trim();
                if (string.IsNullOrWhiteSpace(text)) continue;

                var (top, bot) = _yBounds[bucket];
                var height     = Math.Max(top - bot, 8f);
                lines.Add(new TextLine(text, top, bot, height));
            }
            return lines;
        }
    }
}
