# Intent-Aware Tokenizer

The tokenizer is stage 3 in the MiraV2 pipeline:

```
Transcript (Whisper)
       │
       ▼
  Intent (TAMEV)
       │
       ▼
Tokenizer::process(text, Intent)
       │
       ▼
 TokenResult
       │
       ▼
AppResolver / Tool Resolver (Phase 3)
```

The tokenizer **never** decides or guesses the intent. The `Intent` argument is required and enforced at compile time.

---

## 1. Output Contract (`TokenResult`)

```cpp
enum class TokenType
{
    TARGET,   // Application reference or target name
    QUERY,    // Web search query or screen search terms
    COMMAND,  // System control command or action verb
    TEXT,     // Literal text to type or emit
    PATH,     // File or directory path
    UNKNOWN
};

enum class TokenStatus
{
    Success,           // Valid extraction; value is ready for resolver
    EmptyInput,        // Input utterance was empty or whitespace only
    MissingTarget,     // Recognized command verb had no accompanying target
    UnsupportedIntent  // Intent extractor is not implemented yet (Phase 8)
};

struct TokenResult
{
    TokenType type;
    std::string value;
    TokenStatus status;
    bool valid;        // true iff status == TokenStatus::Success && !value.empty()
};
```

---

## 2. Implemented Intent: `Intent::OpenApplication`

### Contract
- **Input**: Utterance string + `Intent::OpenApplication`.
- **Extraction Goal**: The application name/alias intended to be launched.
- **Rules**:
  1. Trim surrounding whitespace from utterance.
  2. If empty $\to$ `TokenStatus::EmptyInput`, `valid = false`.
  3. Case-insensitive command verbs checked: `"open"`, `"launch"`, `"start"`, `"run"`.
  4. If utterance is a bare command verb without target $\to$ `TokenStatus::MissingTarget`, `valid = false`.
  5. If utterance begins with a verb and space (e.g. `"open   firefox"`), strip verb prefix and any immediate whitespace.
  6. **Preserve** application casing and punctuation (e.g. `"Unity Editor"` stays `"Unity Editor"`, `"7-Zip"` stays `"7-Zip"`, `"Firefox!"` stays `"Firefox!"`). Normalization happens later in the resolver.
  7. If target contains command verbs as part of its name (e.g. `"openoffice"`, `"steam runtime"`), target is preserved intact.

---

## 3. Specification Contracts for Remaining Intents (ROADMAP Phase 8)

The remaining 6 intents are defined below as design contracts. They are not implemented in Phase 2; calls for these intents return `TokenStatus::UnsupportedIntent` with `valid = false`.

### 3.1 `Intent::SearchWeb`
- **Output Token Type**: `TokenType::QUERY`
- **Extraction Rule**:
  - Command verb prefixes stripped case-insensitively: `"search web for "`, `"search for "`, `"google "`, `"look up "`, `"find "`.
  - Punctuation: Strip trailing question marks (e.g. `"how old is linux?"` $\to$ `"how old is linux"`).
  - Empty query or bare verb yields `TokenStatus::MissingTarget`.

### 3.2 `Intent::ReadScreen`
- **Output Token Type**: `TokenType::QUERY` (or `TokenType::COMMAND`)
- **Extraction Rule**:
  - Full screen reads: utterances like `"read screen"`, `"what's on my screen"`, `"summarize screen"` extract an empty or `"all"` query.
  - Targeted queries: utterances like `"find the error on screen"` extract target query `"error"`.
  - Distinguishes whole-display capture from targeted OCR searches.

### 3.3 `Intent::TypeText`
- **Output Token Type**: `TokenType::TEXT`
- **Extraction Rule**:
  - Prefix stripped case-insensitively: `"type "`, `"write "`, `"input "`, `"enter "`.
  - Exact literal string preserved, including casing, spaces, and punctuation.
  - Bare `"type"` yields `TokenStatus::MissingTarget`.

### 3.4 `Intent::FileOperation`
- **Output Token Type**: `TokenType::PATH`
- **Extraction Rule**:
  - Action verb and target path separated: e.g. `"open folder ~/Documents"`, `"delete /tmp/scratch.txt"`.
  - Resolves tilde `~` to `$HOME` and resolves relative paths relative to working directory.

### 3.5 `Intent::SystemControl`
- **Output Token Type**: `TokenType::COMMAND`
- **Extraction Rule**:
  - Canonical verbs extracted: `"lock"`, `"shutdown"`, `"reboot"`, `"volume up"`, `"volume down"`, `"mute"`.
  - Maps natural language variations into standardized action strings.

### 3.6 `Intent::Conversation`
- **Output Token Type**: `TokenType::TEXT`
- **Extraction Rule**:
  - Full utterance preserved as prompt/message for conversational LLM response.
  - Leading assistant trigger words or greetings removed.
